import gradio as gr
import subprocess
from PIL import Image
import os
import time
import uuid
import threading
import re
from pathlib import Path
import tempfile

# ============================================
# DYNAMIC PATHS — all relative to this script
# (equivalent to %~dp0\ggf-zimage\... in batch)
# ============================================
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT    = _BASE_DIR

SD_CLI_PATH  = os.path.join(_APP_ROOT, "sd-cli.exe")
MODEL_DIR    = os.path.join(_APP_ROOT, "models")
GGUF_DIR     = os.path.join(_APP_ROOT, "models", "gguf")
OUTPUT_DIR   = os.path.join(_APP_ROOT, "outputs")
LORA_DIR     = os.path.join(_APP_ROOT, "models", "loras")
VAE_PATH     = os.path.join(_APP_ROOT, "models", "ae.safetensors")

# LLM used by all GGUF diffusion models
GGUF_LLM     = os.path.join(_APP_ROOT, "models", "Qwen3-4B-128K-UD-Q8_K_XL.gguf")
# ============================================

# Directory this script lives in — used for "open explorer" button
SCRIPT_DIR = _BASE_DIR

# ── Static model profiles (bf16 safetensors) ─────────────────────────────────
STATIC_PROFILES = {
    "Turbo bf16": {
        "diffusion_model": os.path.join(MODEL_DIR, "z_image_turbo_bf16.safetensors"),
        "clip_l":          os.path.join(MODEL_DIR, "qwen_3_4b.safetensors"),
        "llm":             os.path.join(MODEL_DIR, "qwen_3_4b.safetensors"),
        "extra_flags":     [],
        "defaults": {
            "steps":           9,
            "cfg_scale":       1.0,
            "width":           1024,
            "height":          720,
            "sampling_method": "",
            "scheduler":       "",
            "flow_shift":      0.0,
            "type_flag":       "",
            "prediction":      "",
        },
    },
    "Base bf16": {
        "diffusion_model": os.path.join(MODEL_DIR, "z_image_bf16.safetensors"),
        "clip_l":          "",
        "llm":             os.path.join(MODEL_DIR, "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
        "extra_flags":     [],
        "defaults": {
            "steps":           25,
            "cfg_scale":       4.0,
            "width":           1024,
            "height":          1024,
            "sampling_method": "euler",
            "scheduler":       "simple",
            "flow_shift":      3.0,
            "type_flag":       "",
            "prediction":      "",
        },
    },
}

# ── Defaults applied to every auto-discovered GGUF ───────────────────────────
GGUF_DEFAULTS = {
    "steps":           9,
    "cfg_scale":       1.0,
    "width":           1024,
    "height":          1024,
    "sampling_method": "euler",
    "scheduler":       "simple",
    "flow_shift":      3.0,
    "type_flag":       "f32",
    "prediction":      "sd3_flow",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LORA_DIR,   exist_ok=True)
os.makedirs(GGUF_DIR,   exist_ok=True)


# ── Profile builders ──────────────────────────────────────────────────────────
def scan_gguf_profiles():
    """One profile per .gguf file in GGUF_DIR, labelled by filename."""
    profiles = {}
    if not os.path.isdir(GGUF_DIR):
        return profiles
    for fname in sorted(os.listdir(GGUF_DIR)):
        if Path(fname).suffix.lower() != ".gguf":
            continue
        label = f"GGUF: {Path(fname).stem}"
        profiles[label] = {
            "diffusion_model": os.path.join(GGUF_DIR, fname),
            "clip_l":          "",
            "llm":             GGUF_LLM,
            "extra_flags":     [],
            "defaults":        dict(GGUF_DEFAULTS),
        }
    return profiles


def build_all_profiles():
    p = dict(STATIC_PROFILES)
    p.update(scan_gguf_profiles())
    return p


# ── LoRA discovery ────────────────────────────────────────────────────────────
def get_lora_list():
    exts = {".safetensors", ".gguf", ".pt", ".bin"}
    if not os.path.isdir(LORA_DIR):
        return []
    return sorted(f for f in os.listdir(LORA_DIR) if Path(f).suffix.lower() in exts)


# ── Model file scanner (for management tab) ───────────────────────────────────
def scan_model_files():
    """Returns dict with lists of files found in each model directory."""
    result = {
        "bf16_models": [],
        "gguf_models": [],
        "loras": [],
        "other": [],
    }

    # Scan main MODEL_DIR for bf16 safetensors
    if os.path.isdir(MODEL_DIR):
        for fname in sorted(os.listdir(MODEL_DIR)):
            fpath = os.path.join(MODEL_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            ext = Path(fname).suffix.lower()
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            if ext == ".safetensors":
                result["bf16_models"].append((fname, size_mb))
            elif ext == ".gguf":
                result["other"].append((fname, size_mb))

    # Scan GGUF_DIR
    if os.path.isdir(GGUF_DIR):
        for fname in sorted(os.listdir(GGUF_DIR)):
            fpath = os.path.join(GGUF_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            ext = Path(fname).suffix.lower()
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            if ext == ".gguf":
                result["gguf_models"].append((fname, size_mb))

    # Scan LORA_DIR
    exts = {".safetensors", ".gguf", ".pt", ".bin"}
    if os.path.isdir(LORA_DIR):
        for fname in sorted(os.listdir(LORA_DIR)):
            fpath = os.path.join(LORA_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            ext = Path(fname).suffix.lower()
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            if ext in exts:
                result["loras"].append((fname, size_mb))

    return result


def build_model_table_html():
    """Build an HTML table showing all discovered model files."""
    data = scan_model_files()

    def rows(items, category, directory):
        if not items:
            return (
                f'<tr><td colspan="4" style="color:#888;font-style:italic;'
                f'padding:6px 12px;">No files found in {directory}</td></tr>'
            )
        out = ""
        for fname, size_mb in items:
            size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.2f} GB"
            out += (
                f'<tr>'
                f'<td style="padding:6px 12px;">{fname}</td>'
                f'<td style="padding:6px 12px;color:#aaa;">{category}</td>'
                f'<td style="padding:6px 12px;color:#aaa;">{size_str}</td>'
                f'<td style="padding:6px 12px;color:#aaa;">{directory}</td>'
                f'</tr>'
            )
        return out

    total_files = sum(len(v) for v in data.values())

    html = (
        '<div style="font-family:monospace;font-size:13px;">'
        '<table style="width:100%;border-collapse:collapse;background:#1a1a1a;border-radius:8px;overflow:hidden;">'
        '<thead>'
        '<tr style="background:#2a2a2a;color:#e0e0e0;">'
        '<th style="padding:10px 12px;text-align:left;">Filename</th>'
        '<th style="padding:10px 12px;text-align:left;">Type</th>'
        '<th style="padding:10px 12px;text-align:left;">Size</th>'
        '<th style="padding:10px 12px;text-align:left;">Directory</th>'
        '</tr>'
        '</thead>'
        '<tbody style="color:#ccc;">'
        + rows(data["bf16_models"], "bf16 / safetensors", MODEL_DIR)
        + rows(data["gguf_models"], "GGUF diffusion", GGUF_DIR)
        + rows(data["loras"], "LoRA", LORA_DIR)
        + rows(data["other"], "other", MODEL_DIR)
        + '</tbody></table>'
        f'<div style="margin-top:8px;color:#888;font-size:12px;">'
        f'{total_files} file(s) found across all model directories.</div>'
        '</div>'
    )
    return html


def open_explorer_to_script_dir():
    try:
        subprocess.Popen(["explorer", SCRIPT_DIR])
        return f"Opened Explorer to: {SCRIPT_DIR}"
    except Exception as e:
        return f"Could not open Explorer: {e}"


def open_explorer_to_model_dir():
    try:
        subprocess.Popen(["explorer", MODEL_DIR])
        return f"Opened Explorer to: {MODEL_DIR}"
    except Exception as e:
        return f"Could not open Explorer: {e}"


def open_explorer_to_gguf_dir():
    try:
        subprocess.Popen(["explorer", GGUF_DIR])
        return f"Opened Explorer to: {GGUF_DIR}"
    except Exception as e:
        return f"Could not open Explorer: {e}"


def open_explorer_to_lora_dir():
    try:
        subprocess.Popen(["explorer", LORA_DIR])
        return f"Opened Explorer to: {LORA_DIR}"
    except Exception as e:
        return f"Could not open Explorer: {e}"


# ── Generation engine ─────────────────────────────────────────────────────────
class ModelManager:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()

    def generate(self, profile, prompt, width, height, steps, cfg_scale, seed,
                 negative_prompt="", selected_loras=None, lora_strength=0.8,
                 sampling_method="", scheduler="", flow_shift=0.0,
                 type_flag="", prediction=""):

        timestamp   = int(time.time())
        output_file = os.path.join(OUTPUT_DIR, f"zimage_{timestamp}_{uuid.uuid4().hex[:8]}.png")

        full_prompt = prompt
        if selected_loras:
            for lf in selected_loras:
                full_prompt += f" <lora:{Path(lf).stem}:{lora_strength:.2f}>"

        cmd = [SD_CLI_PATH, "--diffusion-model", profile["diffusion_model"]]

        if profile["clip_l"]:
            cmd += ["--clip_l", profile["clip_l"]]

        cmd += ["--llm", profile["llm"], "--vae", VAE_PATH]

        if type_flag:
            cmd += ["--type", type_flag]

        cmd += ["--lora-model-dir", LORA_DIR]
        cmd += ["-p", full_prompt]

        if negative_prompt and negative_prompt.strip():
            cmd += ["--negative-prompt", negative_prompt]

        cmd += ["-W", str(int(width)), "-H", str(int(height))]
        cmd += ["--steps", str(steps), "--cfg-scale", str(cfg_scale)]

        if sampling_method:
            cmd += ["--sampling-method", sampling_method]
        if scheduler:
            cmd += ["--scheduler", scheduler]
        if flow_shift and float(flow_shift) != 0.0:
            cmd += ["--flow-shift", str(flow_shift)]
        if prediction:
            cmd += ["--prediction", prediction]

        cmd += ["--seed", str(seed)]
        cmd += ["-o", output_file]
        cmd += profile.get("extra_flags", [])

        print("Running:", " ".join(cmd))

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        step_pat = re.compile(r"\s*(\d+)/(\d+)\s*-\s*[0-9.]+s/it")

        for line in process.stdout:
            print(line.rstrip())
            m = step_pat.search(line)
            if m:
                yield None, None, f"Generating: Step {m.group(1)}/{m.group(2)}...", ""

        process.wait()

        if os.path.exists(output_file):
            img = Image.open(output_file)
            yield img, output_file, f"Done in {time.time()-timestamp:.2f}s", output_file
        else:
            yield None, None, "Error: image not generated", ""


# ── Model info markdown ───────────────────────────────────────────────────────
def _build_model_info(model_name, profiles):
    p = profiles[model_name]
    d = p["defaults"]
    lines = [
        f"**Active profile:** `{model_name}`",
        f"- **Diffusion model:** `{os.path.basename(p['diffusion_model'])}`",
        f"- **LLM:** `{os.path.basename(p['llm'])}`",
    ]
    if p["clip_l"]:
        lines.append(f"- **CLIP-L:** `{os.path.basename(p['clip_l'])}`")
    lines += [
        f"- **VAE:** `{os.path.basename(VAE_PATH)}`",
        f"- **--type:** `{d['type_flag'] or 'omitted'}`",
        f"- **--prediction:** `{d['prediction'] or 'omitted'}`",
        f"- **Default steps / CFG:** `{d['steps']} / {d['cfg_scale']}`",
        f"- **Sampling / scheduler:** `{d['sampling_method'] or 'default'}` / `{d['scheduler'] or 'default'}`",
        f"- **Flow shift:** `{d['flow_shift'] or 'omitted'}`",
        f"- **GGUF dir:** `{GGUF_DIR}`",
        f"- **LoRA dir:** `{LORA_DIR}`",
        f"- **Output dir:** `{OUTPUT_DIR}`",
        f"- **sd-cli:** `{SD_CLI_PATH}`",
    ]
    return "\n".join(lines)


# ── UI ────────────────────────────────────────────────────────────────────────
def create_ui():
    import random as _rnd

    manager  = ModelManager()
    profiles = build_all_profiles()
    names    = list(profiles.keys())

    with gr.Blocks(title="Z-Image Generator") as demo:

        _profiles_state = gr.State(profiles)

        gr.Markdown(
            "## Z-Image Generator  \n"
            "**Ctrl+Enter** to generate &nbsp;|&nbsp; "
            f"GGUF dir: `{GGUF_DIR}`  \n"
            "Find more A.I. apps at [Get Going Fast](https://getgoingfast.pro)"
        )

        gr.HTML("""
        <script>
        (function() {
            function fixFullscreen() {
                try {
                    const img = document.querySelector('#component-output-img img');
                    if (!img) return;
                    const toolbar = document.querySelector('[data-testid="image"] .image-toolbar');
                    if (!toolbar) return;
                    const buttons = toolbar.querySelectorAll('button');
                    const fullscreenBtn = buttons[2] || buttons[buttons.length-1];
                    if (fullscreenBtn) {
                        fullscreenBtn.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            window.open(img.src, '_blank');
                            return false;
                        };
                    }
                } catch (e) {
                    console.log('Fullscreen fix error:', e);
                }
            }
            const target = document.getElementById('component-output-img');
            if (target) {
                const observer = new MutationObserver(fixFullscreen);
                observer.observe(target, { childList: true, subtree: true });
                setTimeout(fixFullscreen, 500);
            }
        })();
        </script>
        """)

        # ── TABS ──────────────────────────────────────────────────────────────
        with gr.Tabs():

            # ════════════════════════════════════════════════════════════════
            # TAB 1 — Generate
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("🎨 Generate"):
                with gr.Row():
                    # ── Controls ──────────────────────────────────────────────
                    with gr.Column(scale=2):

                        with gr.Row():
                            model_select = gr.Dropdown(
                                choices=names,
                                value=names[0],
                                label="Model",
                                scale=4,
                                info="GGUF models auto-discovered from GGUF dir above",
                            )
                            refresh_models_btn = gr.Button("Refresh Models", scale=1, size="sm")

                        prompt = gr.Textbox(
                            label="Prompt",
                            value="Cinematic scene of four heroic ninja turtles standing on a New York rooftop at sunset, dramatic orange and purple sky, city skyline glowing behind them. One turtle spray-paints a glowing neon graffiti tag on a brick wall that reads 'Get Going Fast'. Another turtle holds a slice of pizza while leaning on an air vent, a third is flirting with Taylor Swift who is dressed as a reporter with a fedora and a camera hanging around he neck, as she leans coyly back against the wall with one leg against the wall and the other on the ground, and the fourth points toward the city like a team leader. Wet rooftop reflecting neon light, steam rising from vents, dynamic comic-book energy mixed with photorealistic textures, ultra detailed, cinematic lighting, depth of field, 4k, sharp focus, epic composition.",
                            lines=3,
                        )
                        negative = gr.Textbox(
                            label="Negative Prompt",
                            value="blurry, low quality, artifacts",
                            lines=2,
                        )

                        with gr.Row():
                            with gr.Column():
                                width = gr.Number(label="Width",  value=1024, minimum=256, maximum=2048, step=64)
                                steps = gr.Slider(label="Steps",  minimum=1,  maximum=80,  value=9,      step=1)
                            with gr.Column():
                                height    = gr.Number(label="Height",    value=1024, minimum=256, maximum=2048, step=64)
                                cfg_scale = gr.Slider(label="CFG Scale", minimum=0,  maximum=20,  value=1.0,    step=0.1)

                        with gr.Row():
                            seed            = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                            random_seed_btn = gr.Button("Random Seed")

                        with gr.Accordion("Advanced Settings", open=False) as adv_accordion:
                            gr.Markdown(
                                "_Auto-populated when model changes. "
                                "Empty dropdowns = flag omitted from command._"
                            )
                            with gr.Row():
                                sampling_method = gr.Dropdown(
                                    choices=["", "euler", "euler_a", "dpm2", "dpm++2m",
                                             "dpm++2mv2", "lcm", "ddim"],
                                    value="",
                                    label="--sampling-method",
                                )
                                scheduler = gr.Dropdown(
                                    choices=["", "simple", "discrete", "karras",
                                             "exponential", "ays", "gits"],
                                    value="",
                                    label="--scheduler",
                                )
                            with gr.Row():
                                flow_shift = gr.Slider(
                                    label="--flow-shift  (0 = omit)",
                                    minimum=0.0, maximum=10.0, value=0.0, step=0.5,
                                )
                                type_flag = gr.Dropdown(
                                    choices=["", "f32", "f16", "bf16", "q8_0", "q4_0"],
                                    value="",
                                    label="--type  (blank = omit)",
                                )
                            prediction = gr.Dropdown(
                                choices=["", "sd3_flow", "eps", "v", "flow"],
                                value="",
                                label="--prediction  (blank = omit)",
                            )

                        with gr.Accordion("LoRA Models", open=False):
                            lora_select = gr.CheckboxGroup(
                                choices=get_lora_list(),
                                label=f"Available LoRAs  ({LORA_DIR})",
                                value=[],
                            )
                            lora_strength     = gr.Slider(label="LoRA Strength", minimum=0.0, maximum=2.0, value=0.8, step=0.05)
                            refresh_loras_btn = gr.Button("Refresh LoRA List")

                        with gr.Row():
                            generate_btn = gr.Button("Generate", variant="primary", size="lg", elem_id="generate-btn")
                            clear_btn    = gr.Button("Clear")

                    # ── Output ────────────────────────────────────────────────
                    with gr.Column(scale=3):
                        output_img = gr.Image(
                            label="Generated Image",
                            type="pil",
                            interactive=False,
                            height=520,
                            elem_id="component-output-img",
                        )
                        status    = gr.Textbox(label="Status",   interactive=False)
                        save_path = gr.Textbox(label="Saved to", interactive=False)

                with gr.Accordion("Model Info", open=False):
                    model_info_md = gr.Markdown(_build_model_info(names[0], profiles))

            # ════════════════════════════════════════════════════════════════
            # TAB 2 — Add / Remove Models
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("📦 Add / Remove Models"):

                gr.HTML("""
                <div style="
                    background: linear-gradient(135deg, #1e1e2e 0%, #2a1a0e 100%);
                    border: 1px solid #f59e0b;
                    border-radius: 10px;
                    padding: 20px 24px;
                    margin-bottom: 16px;
                ">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                        <span style="font-size:24px;">&#9888;&#65039;</span>
                        <span style="font-size:16px;font-weight:700;color:#f59e0b;font-family:monospace;">
                            Downloading &amp; deleting models requires the Model Manager batch script
                        </span>
                    </div>
                    <p style="color:#d1c5a0;font-size:14px;margin:0 0 10px 0;line-height:1.6;">
                        Browsers cannot download multi-gigabyte model files or delete files from your drive.
                        Use the <strong style="color:#fde68a;">Model Manager .bat</strong> file that came
                        with this app &mdash; it handles downloads, integrity checks, and safe deletion
                        with a guided menu.
                    </p>
                    <p style="color:#d1c5a0;font-size:14px;margin:0;line-height:1.6;">
                        Click <strong style="color:#fde68a;">Open App Folder</strong> below to jump straight
                        to the directory &mdash; look for
                        <code style="background:#3a2a0a;padding:2px 6px;border-radius:4px;">
                        Z-Image-Model-Manager.bat</code> and double-click to run it.
                    </p>
                </div>
                """)

                # ── Explorer quick-launch buttons ─────────────────────────────
                gr.Markdown("### Open Folder in Explorer")
                with gr.Row():
                    open_script_btn = gr.Button("📁 Open App Folder",    variant="primary",   size="lg")
                    open_model_btn  = gr.Button("📁 Models Folder",       variant="secondary", size="sm")
                    open_gguf_btn   = gr.Button("📁 GGUF Folder",         variant="secondary", size="sm")
                    open_lora_btn   = gr.Button("📁 LoRA Folder",         variant="secondary", size="sm")

                explorer_status = gr.Textbox(
                    label="Explorer Status",
                    interactive=False,
                    value="",
                    max_lines=1,
                )

                gr.Markdown("---")

                # ── Current model inventory ───────────────────────────────────
                gr.Markdown("### Current Model Inventory")
                gr.Markdown(
                    "_Read-only view of model files currently on disk. "
                    "Use the Model Manager .bat to add or remove files, "
                    "then click **Refresh Inventory** to update this list._"
                )

                refresh_inventory_btn = gr.Button("🔄 Refresh Inventory", variant="secondary")
                model_inventory_html  = gr.HTML(build_model_table_html())

                gr.Markdown("---")

                # ── Directory reference ───────────────────────────────────────
                gr.Markdown("### Configured Directories")
                gr.Markdown(
                    f"| Directory | Path |\n"
                    f"|-----------|------|\n"
                    f"| App / Script | `{SCRIPT_DIR}` |\n"
                    f"| Models (bf16) | `{MODEL_DIR}` |\n"
                    f"| GGUF Diffusion | `{GGUF_DIR}` |\n"
                    f"| LoRAs | `{LORA_DIR}` |\n"
                    f"| Outputs | `{OUTPUT_DIR}` |\n"
                )

        # ── Callbacks ─────────────────────────────────────────────────────────
        def on_model_change(model_name, current_profiles):
            if model_name not in current_profiles:
                return (gr.update(),) * 11
            d = current_profiles[model_name]["defaults"]
            show_adv = bool(d["sampling_method"] or d["scheduler"]
                            or d["flow_shift"] or d["type_flag"] or d["prediction"])
            return (
                d["steps"],
                d["cfg_scale"],
                d["width"],
                d["height"],
                d["sampling_method"] or "",
                d["scheduler"]       or "",
                d["flow_shift"]      if d["flow_shift"] else 0.0,
                d["type_flag"]       or "",
                d["prediction"]      or "",
                gr.Accordion(open=show_adv),
                _build_model_info(model_name, current_profiles),
            )

        def refresh_models_fn(current_profiles):
            new_profiles = build_all_profiles()
            new_names    = list(new_profiles.keys())
            first        = new_names[0]
            return (
                gr.Dropdown(choices=new_names, value=first),
                new_profiles,
                _build_model_info(first, new_profiles),
            )

        def generate_wrapper(model_name, current_profiles,
                             prompt_val, neg, w, h, steps_val, cfg, seed_val,
                             sel_loras, lora_str,
                             samp_method, sched, f_shift, t_flag, pred):
            if seed_val == -1:
                seed_val = _rnd.randint(1, 999999)
                yield None, f"Using random seed: {seed_val}", ""

            profile = current_profiles[model_name]
            for preview, final_path, status_msg, sp in manager.generate(
                profile, prompt_val, w, h, steps_val, cfg, seed_val, neg,
                selected_loras=sel_loras, lora_strength=lora_str,
                sampling_method=samp_method, scheduler=sched,
                flow_shift=f_shift, type_flag=t_flag, prediction=pred,
            ):
                yield (final_path if final_path else preview), status_msg, sp

        # ── wire Generate tab events ───────────────────────────────────────────
        model_select.change(
            fn=on_model_change,
            inputs=[model_select, _profiles_state],
            outputs=[steps, cfg_scale, width, height,
                     sampling_method, scheduler, flow_shift,
                     type_flag, prediction,
                     adv_accordion, model_info_md],
        )
        refresh_models_btn.click(
            fn=refresh_models_fn,
            inputs=[_profiles_state],
            outputs=[model_select, _profiles_state, model_info_md],
        )
        random_seed_btn.click(fn=lambda: _rnd.randint(1, 999999), outputs=[seed])
        clear_btn.click(fn=lambda: (None, "", ""), outputs=[output_img, status, save_path])
        refresh_loras_btn.click(
            fn=lambda: gr.CheckboxGroup(choices=get_lora_list(), value=[]),
            outputs=[lora_select],
        )
        generate_btn.click(
            fn=generate_wrapper,
            inputs=[model_select, _profiles_state,
                    prompt, negative, width, height, steps, cfg_scale, seed,
                    lora_select, lora_strength,
                    sampling_method, scheduler, flow_shift, type_flag, prediction],
            outputs=[output_img, status, save_path],
        )

        # ── wire Model Management tab events ──────────────────────────────────
        open_script_btn.click(fn=open_explorer_to_script_dir, outputs=[explorer_status])
        open_model_btn.click( fn=open_explorer_to_model_dir,  outputs=[explorer_status])
        open_gguf_btn.click(  fn=open_explorer_to_gguf_dir,   outputs=[explorer_status])
        open_lora_btn.click(  fn=open_explorer_to_lora_dir,   outputs=[explorer_status])
        refresh_inventory_btn.click(fn=build_model_table_html, outputs=[model_inventory_html])

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        js="""
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                const btn = document.getElementById('generate-btn');
                if (btn) btn.click();
            }
        });
        """
    )