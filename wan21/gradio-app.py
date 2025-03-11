import os
import sys
import subprocess
import random
import argparse
import time
import tempfile
import json

import torch
import gradio as gr
from PIL import Image
import cv2

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video

from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download

# ------------------------------
# Helper function for common model files
# ------------------------------
def get_common_file(new_path, old_path):
    if os.path.exists(new_path):
        return new_path
    elif os.path.exists(old_path):
        return old_path
    else:
        print(f"[WARNING] Neither {new_path} nor {old_path} found. Using {old_path} as fallback.")
        return old_path

# ------------------------------
# Config management functions
# ------------------------------
CONFIG_DIR = "configs"
LAST_CONFIG_FILE = os.path.join(CONFIG_DIR, "last_used_config.txt")
DEFAULT_CONFIG_NAME = "Default"

def get_default_config():
    return {
        "model_choice": "WAN 2.1 1.3B Text-to-Video",
        "vram_preset": "24GB",
        "aspect_ratio": "16:9",
        "width": 832,
        "height": 480,
        "auto_crop": True,
        "tiled": True,
        "inference_steps": 50,
        "pr_rife": True,
        "pr_rife_multiplier": "2x FPS",
        "cfg_scale": 6.0,
        "sigma_shift": 6.0,
        "num_persistent": "12000000000",
        "torch_dtype": "torch.bfloat16",
        "lora_model": "None",
        "lora_alpha": 1.0,
        "negative_prompt": "still and motionless picture, static",
        "save_prompt": True,
        "multiline": False,
        "num_generations": 1,
        "use_random_seed": True,
        "seed": "",
        "quality": 5,
        "fps": 16,
        "num_frames": 81,
        "tar_lang": "EN"
    }

if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

default_config = get_default_config()

if os.path.exists(LAST_CONFIG_FILE):
    with open(LAST_CONFIG_FILE, "r", encoding="utf-8") as f:
        last_config_name = f.read().strip()
    config_file_path = os.path.join(CONFIG_DIR, f"{last_config_name}.json")
    if os.path.exists(config_file_path):
        with open(config_file_path, "r", encoding="utf-8") as f:
            config_loaded = json.load(f)
        last_config = last_config_name
    else:
        default_config_path = os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")
        if os.path.exists(default_config_path):
            with open(default_config_path, "r", encoding="utf-8") as f:
                config_loaded = json.load(f)
            last_config = DEFAULT_CONFIG_NAME
        else:
            config_loaded = default_config
            with open(default_config_path, "w", encoding="utf-8") as f:
                json.dump(config_loaded, f, indent=4)
            last_config = DEFAULT_CONFIG_NAME
            with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
                f.write(last_config)
else:
    default_config_path = os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")
    if os.path.exists(default_config_path):
        with open(default_config_path, "r", encoding="utf-8") as f:
            config_loaded = json.load(f)
        last_config = DEFAULT_CONFIG_NAME
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_NAME)
    else:
        config_loaded = default_config
        with open(default_config_path, "w", encoding="utf-8") as f:
            json.dump(config_loaded, f, indent=4)
        last_config = DEFAULT_CONFIG_NAME
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_NAME)

def get_config_list():
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
    files = os.listdir(CONFIG_DIR)
    configs = [os.path.splitext(f)[0] for f in files if f.endswith(".json")]
    return sorted(configs)

def save_config(config_name, model_choice, vram_preset, aspect_ratio, width, height, auto_crop, tiled, inference_steps,
                pr_rife, pr_rife_multiplier, cfg_scale, sigma_shift, num_persistent, torch_dtype, lora_model, lora_alpha,
                negative_prompt, save_prompt, multiline, num_generations, use_random_seed, seed, quality, fps, num_frames,
                tar_lang):
    if not config_name:
        return "Config name cannot be empty", gr.update(choices=get_config_list())
    config_data = {
        "model_choice": model_choice,
        "vram_preset": vram_preset,
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "auto_crop": auto_crop,
        "tiled": tiled,
        "inference_steps": inference_steps,
        "pr_rife": pr_rife,
        "pr_rife_multiplier": pr_rife_multiplier,
        "cfg_scale": cfg_scale,
        "sigma_shift": sigma_shift,
        "num_persistent": num_persistent,
        "torch_dtype": torch_dtype,
        "lora_model": lora_model,
        "lora_alpha": lora_alpha,
        "negative_prompt": negative_prompt,
        "save_prompt": save_prompt,
        "multiline": multiline,
        "num_generations": num_generations,
        "use_random_seed": use_random_seed,
        "seed": seed,
        "quality": quality,
        "fps": fps,
        "num_frames": num_frames,
        "tar_lang": tar_lang
    }
    config_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)
    with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(config_name)
    return f"Config '{config_name}' saved.", gr.update(choices=get_config_list(), value=config_name)

def load_config(selected_config):
    config_path = os.path.join(CONFIG_DIR, f"{selected_config}.json")
    if not os.path.exists(config_path):
        default_vals = get_default_config()
        return (f"Config '{selected_config}' not found.",
                default_vals["model_choice"],
                default_vals["vram_preset"],
                default_vals["aspect_ratio"],
                default_vals["width"],
                default_vals["height"],
                default_vals["auto_crop"],
                default_vals["tiled"],
                default_vals["inference_steps"],
                default_vals["pr_rife"],
                default_vals["pr_rife_multiplier"],
                default_vals["cfg_scale"],
                default_vals["sigma_shift"],
                default_vals["num_persistent"],
                default_vals["torch_dtype"],
                default_vals["lora_model"],
                default_vals["lora_alpha"],
                default_vals["negative_prompt"],
                default_vals["save_prompt"],
                default_vals["multiline"],
                default_vals["num_generations"],
                default_vals["use_random_seed"],
                default_vals["seed"],
                default_vals["quality"],
                default_vals["fps"],
                default_vals["num_frames"],
                default_vals["tar_lang"],
                "" )
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(selected_config)
    return (f"Config '{selected_config}' loaded.",
            config_data.get("model_choice", "WAN 2.1 1.3B Text-to-Video"),
            config_data.get("vram_preset", "24GB"),
            config_data.get("aspect_ratio", "16:9"),
            config_data.get("width", 832),
            config_data.get("height", 480),
            config_data.get("auto_crop", True),
            config_data.get("tiled", True),
            config_data.get("inference_steps", 50),
            config_data.get("pr_rife", True),
            config_data.get("pr_rife_multiplier", "2x FPS"),
            config_data.get("cfg_scale", 6.0),
            config_data.get("sigma_shift", 6.0),
            config_data.get("num_persistent", "12000000000"),
            config_data.get("torch_dtype", "torch.bfloat16"),
            config_data.get("lora_model", "None"),
            config_data.get("lora_alpha", 1.0),
            config_data.get("negative_prompt", "still and motionless picture, static"),
            config_data.get("save_prompt", True),
            config_data.get("multiline", False),
            config_data.get("num_generations", 1),
            config_data.get("use_random_seed", True),
            config_data.get("seed", ""),
            config_data.get("quality", 5),
            config_data.get("fps", 16),
            config_data.get("num_frames", 81),
            config_data.get("tar_lang", "EN"),
            "" )

# ------------------------------
# Constants and Helper Functions
# ------------------------------
ASPECT_RATIOS_1_3b = {
    "1:1":  (480, 480),
    "4:3":  (608, 456),
    "3:4":  (456, 608),
    "3:2":  (608, 405),
    "2:3":  (405, 608),
    "16:9": (672, 378),
    "9:16": (378, 672),
    "21:9": (800, 336),
    "9:21": (336, 800),
    "4:5":  (456, 570),
    "5:4":  (570, 456),
}

ASPECT_RATIOS_14b = {
    "1:1":  (832, 832),
    "4:3":  (912, 684),
    "3:4":  (684, 912),
    "3:2":  (960, 640),
    "2:3":  (640, 960),
    "16:9": (1088, 612),
    "16:9_low": (672, 378),
    "9:16": (612, 1088),
    "9:16_low": (378, 672),
    "21:9": (1216, 512),
    "9:21": (512, 1216),
    "4:5":  (684, 852),
    "5:4":  (852, 684),
}

def update_vram_and_resolution(model_choice, preset):
    print(model_choice)
    if model_choice == "WAN 2.1 1.3B Text-to-Video":
        mapping = {
            "4GB": "0",
            "6GB": "500000000",
            "8GB": "1000000000",
            "10GB": "7000000000",
            "12GB": "7000000000",
            "16GB": "7000000000",
            "24GB": "7000000000",
            "32GB": "7000000000"
        }
        resolution_choices = list(ASPECT_RATIOS_1_3b.keys())
        default_aspect = "16:9"
    elif model_choice == "WAN 2.1 14B Text-to-Video":
        mapping = {
            "4GB": "0",
            "6GB": "0",
            "8GB": "0",
            "10GB": "0",
            "12GB": "0",
            "16GB": "0",
            "24GB": "3000000000",
            "32GB": "6500000000"
        }
        resolution_choices = list(ASPECT_RATIOS_14b.keys())
        default_aspect = "16:9"
    elif model_choice == "WAN 2.1 14B Image-to-Video 720P":
        mapping = {
            "4GB": "0",
            "6GB": "0",
            "8GB": "0",
            "10GB": "0",
            "12GB": "0",
            "16GB": "0",
            "24GB": "0",
            "32GB": "3500000000"
        }
        resolution_choices = list(ASPECT_RATIOS_14b.keys())
        default_aspect = "16:9"
    elif model_choice == "WAN 2.1 14B Image-to-Video 480P":
        mapping = {
            "4GB": "0",
            "6GB": "0",
            "8GB": "0",
            "10GB": "0",
            "12GB": "0",
            "16GB": "1200000000",
            "24GB": "5000000000",
            "32GB": "9500000000"
        }
        resolution_choices = list(ASPECT_RATIOS_1_3b.keys())
        default_aspect = "16:9"
    else:
        mapping = {
            "4GB": "0",
            "6GB": "0",
            "8GB": "0",
            "10GB": "0",
            "12GB": "0",
            "16GB": "0",
            "24GB": "0",
            "32GB": "12000000000"
        }
        resolution_choices = list(ASPECT_RATIOS_14b.keys())
        default_aspect = "16:9"
    return mapping.get(preset, "12000000000"), resolution_choices, default_aspect

def update_model_settings(model_choice, current_vram_preset):
    num_persistent_val, aspect_options, default_aspect = update_vram_and_resolution(model_choice, current_vram_preset)
    if model_choice == "WAN 2.1 1.3B Text-to-Video" or model_choice == "WAN 2.1 14B Image-to-Video 480P":
        default_width, default_height = ASPECT_RATIOS_1_3b.get(default_aspect, (832, 480))
    else:
        default_width, default_height = ASPECT_RATIOS_14b.get(default_aspect, (1280, 720))
    return (
        gr.update(choices=aspect_options, value=default_aspect),
        default_width,
        default_height,
        num_persistent_val
    )

def update_width_height(aspect_ratio, model_choice):
    if model_choice == "WAN 2.1 1.3B Text-to-Video" or model_choice == "WAN 2.1 14B Image-to-Video 480P":
        default_width, default_height = ASPECT_RATIOS_1_3b.get(aspect_ratio, (832, 480))
    else:
        default_width, default_height = ASPECT_RATIOS_14b.get(aspect_ratio, (1280, 720))
    return default_width, default_height

def update_vram_on_change(preset, model_choice):
    num_persistent_val, _, _ = update_vram_and_resolution(model_choice, preset)
    return num_persistent_val

def auto_crop_image(image, target_width, target_height):
    w, h = image.size
    target_ratio = target_width / target_height
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_width = int(h * target_ratio)
        left = (w - new_width) // 2
        right = left + new_width
        image = image.crop((left, 0, right, h))
    elif current_ratio < target_ratio:
        new_height = int(w / target_ratio)
        top = (h - new_height) // 2
        bottom = top + new_height
        image = image.crop((0, top, w, bottom))
    image = image.resize((target_width, target_height), Image.LANCZOS)
    return image

def prompt_enc(prompt, tar_lang):
    global prompt_expander, loaded_pipeline, loaded_pipeline_config, args
    if prompt_expander is None:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(model_name=args.prompt_extend_model, is_vl=False)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(model_name=args.prompt_extend_model, is_vl=False, device=0)
        else:
            raise NotImplementedError(f"Unsupported prompt_extend_method: {args.prompt_extend_method}")
    prompt_output = prompt_expander(prompt, tar_lang=tar_lang.lower())
    result = prompt if not prompt_output.status else prompt_output.prompt
    return result

def generate_videos(
    prompt, tar_lang, negative_prompt, input_image, num_generations,
    save_prompt, multi_line, use_random_seed, seed_input, quality, fps,
    model_choice_radio, vram_preset, num_persistent_input, torch_dtype, num_frames,
    aspect_ratio, width, height, auto_crop, tiled, inference_steps, pr_rife_enabled, pr_rife_radio, cfg_scale, sigma_shift,
    lora_model, lora_alpha
):
    global loaded_pipeline, loaded_pipeline_config, cancel_flag
    cancel_flag = False
    log_text = ""
    last_used_seed = None
    last_video_path = ""
    overall_start_time = time.time()

    if model_choice_radio == "WAN 2.1 1.3B Text-to-Video":
        model_choice = "1.3B"
        d = ASPECT_RATIOS_1_3b
    elif model_choice_radio == "WAN 2.1 14B Text-to-Video":
        model_choice = "14B_text"
        d = ASPECT_RATIOS_14b
    elif model_choice_radio == "WAN 2.1 14B Image-to-Video 720P":
        model_choice = "14B_image_720p"
        d = ASPECT_RATIOS_14b
    elif model_choice_radio == "WAN 2.1 14B Image-to-Video 480P":
        model_choice = "14B_image_480p"
        d = ASPECT_RATIOS_1_3b
    else:
        return "", log_text, str(last_used_seed or "")
    
    target_width = int(width)
    target_height = int(height)

    effective_num_frames = int(num_frames)

    if auto_crop and input_image is not None and model_choice in ["14B_image_720p", "14B_image_480p"]:
        original_image = input_image.copy()

    vram_value = num_persistent_input

    if lora_model == "None" or not lora_model:
        effective_lora_model = None
    else:
        effective_lora_model = os.path.join("LoRAs", lora_model)

    current_config = {
        "model_choice": model_choice,
        "torch_dtype": torch_dtype,
        "num_persistent": vram_value,
        "lora_model": effective_lora_model,
        "lora_alpha": lora_alpha,
    }

    if loaded_pipeline is None or loaded_pipeline_config != current_config:
        if effective_lora_model is not None:
            print(f"[CMD] Applying LoRA: {effective_lora_model} with scale {lora_alpha}")
        else:
            print("[CMD] No LoRA selected. Using base model.")
        loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_value,
                                            lora_path=effective_lora_model, lora_alpha=lora_alpha)
        loaded_pipeline_config = current_config

    if multi_line:
        prompts_list = [line.strip() for line in prompt.splitlines() if line.strip()]
    else:
        prompts_list = [prompt.strip()]

    total_iterations = len(prompts_list) * int(num_generations)
    iteration = 0

    for p in prompts_list:
        for i in range(int(num_generations)):
            if cancel_flag:
                log_text += "[CMD] Generation cancelled by user.\n"
                print("[CMD] Generation cancelled by user.")
                duration = time.time() - overall_start_time
                log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
                log_text += f"[CMD] Generation complete. Duration: {duration:.2f} seconds. Last used seed: {last_used_seed}\n"
                loaded_pipeline = None
                loaded_pipeline_config = {}
                return "", log_text, str(last_used_seed or "")
            iteration += 1

            iter_start = time.time()

            log_text += f"[CMD] Generating video {iteration} of {total_iterations} with prompt: {p}\n"
            print(f"[CMD] Generating video {iteration}/{total_iterations} with prompt: {p}")

            enhanced_prompt = p

            if use_random_seed:
                current_seed = random.randint(0, 2**32 - 1)
            else:
                try:
                    current_seed = int(seed_input) if seed_input.strip() != "" else 0
                except Exception as e:
                    current_seed = 0
            last_used_seed = current_seed
            print(f"[CMD] Using resolution: width={target_width}  height={target_height}")

            common_args = {
                "prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": int(inference_steps),
                "seed": current_seed,
                "tiled": tiled,
                "width": target_width,
                "height": target_height,
                "num_frames": effective_num_frames,
                "cfg_scale": cfg_scale,
                "sigma_shift": sigma_shift,
            }

            if model_choice == "1.3B":
                video_data = loaded_pipeline(**common_args)
                video_filename = get_next_filename(".mp4")
            elif model_choice == "14B_text":
                video_data = loaded_pipeline(**common_args)
                video_filename = get_next_filename(".mp4")
            elif model_choice in ["14B_image_720p", "14B_image_480p"]:
                if input_image is None:
                    err_msg = "[CMD] Error: Image model selected but no image provided."
                    print(err_msg)
                    loaded_pipeline = None
                    loaded_pipeline_config = {}
                    return "", err_msg, str(last_used_seed or "")
                if auto_crop:
                    processed_image = auto_crop_image(original_image, target_width, target_height)
                else:
                    processed_image = original_image
                video_filename = get_next_filename(".mp4")
                preprocessed_folder = "auto_pre_processed_images"
                if not os.path.exists(preprocessed_folder):
                    os.makedirs(preprocessed_folder)
                base_name = os.path.splitext(os.path.basename(video_filename))[0]
                preprocessed_image_filename = os.path.join(preprocessed_folder, f"{base_name}.png")
                processed_image.save(preprocessed_image_filename)
                log_text += f"[CMD] Saved auto processed image: {preprocessed_image_filename}\n"
                print(f"[CMD] Saved auto processed image: {preprocessed_image_filename}")
                video_data = loaded_pipeline(input_image=processed_image, **common_args)
            else:
                err_msg = "[CMD] Invalid combination of inputs."
                print(err_msg)
                loaded_pipeline = None
                loaded_pipeline_config = {}
                return "", err_msg, str(last_used_seed or "")

            save_video(video_data, video_filename, fps=fps, quality=quality)
            log_text += f"[CMD] Saved video: {video_filename}\n"
            print(f"[CMD] Saved video: {video_filename}")

            if save_prompt:
                text_filename = os.path.splitext(video_filename)[0] + ".txt"
                generation_details = ""
                generation_details += f"Prompt: {enhanced_prompt}\n"
                generation_details += f"Negative Prompt: {negative_prompt}\n"
                generation_details += f"Used Model: {model_choice_radio}\n"
                generation_details += f"Number of Inference Steps: {inference_steps}\n"
                generation_details += f"Seed: {current_seed}\n"
                generation_details += f"Number of Frames: {effective_num_frames}\n"
                generation_details += "Denoising Strength: N/A\n"
                if lora_model and lora_model != "None":
                    generation_details += f"LoRA Model: {lora_model} with scale {lora_alpha}\n"
                else:
                    generation_details += "LoRA Model: None\n"
                generation_details += f"Precision: {'FP8' if torch_dtype == 'torch.float8_e4m3fn' else 'BF16'}\n"
                generation_details += f"Auto Crop: {'Enabled' if auto_crop else 'Disabled'}\n"
                generation_details += f"Generation Duration: {time.time()-iter_start:.2f} seconds / {(time.time()-iter_start)/60:.2f} minutes\n"
                with open(text_filename, "w", encoding="utf-8") as f:
                    f.write(generation_details)
                log_text += f"[CMD] Saved prompt and parameters: {text_filename}\n"
                print(f"[CMD] Saved prompt and parameters: {text_filename}")

            last_video_path = video_filename

    if pr_rife_enabled and last_video_path:
        print(f"[CMD] Applying Practical-RIFE with multiplier {pr_rife_radio} on video {last_video_path}")
        multiplier_val = "2" if pr_rife_radio == "2x FPS" else "4"
        improved_video = os.path.join("outputs", "improved_" + os.path.basename(last_video_path))
        model_dir = os.path.abspath(os.path.join("Practical-RIFE", "train_log"))
        cmd = (
            f'"{sys.executable}" "Practical-RIFE/inference_video.py" '
            f'--model="{model_dir}" --multi={multiplier_val} '
            f'--video="{last_video_path}" --output="{improved_video}"'
        )
        print(f"[CMD] Running command: {cmd}")
        subprocess.run(cmd, shell=True, check=True, env=os.environ)
        print(f"[CMD] Practical-RIFE finished. Improved video saved to: {improved_video}")
        last_video_path = improved_video
        log_text += f"[CMD] Applied Practical-RIFE with multiplier {multiplier_val}x. Improved video saved to {improved_video}\n"

    overall_duration = time.time() - overall_start_time
    log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
    log_text += f"[CMD] Generation complete. Overall Duration: {overall_duration:.2f} seconds ({overall_duration/60:.2f} minutes). Last used seed: {last_used_seed}\n"
    print(f"[CMD] Generation complete. Overall Duration: {overall_duration:.2f} seconds. Last used seed: {last_used_seed}")

    loaded_pipeline = None
    loaded_pipeline_config = {}

    return last_video_path, log_text, str(last_used_seed or "")

def cancel_generation():
    global cancel_flag
    cancel_flag = True
    print("[CMD] Cancel button pressed.")
    return "Cancelling generation..."

def get_next_filename(extension):
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    existing_files = [f for f in os.listdir(outputs_dir) if f.endswith(extension)]
    current_numbers = []
    for file in existing_files:
        try:
            num = int(os.path.splitext(file)[0])
            current_numbers.append(num)
        except Exception as e:
            continue
    next_number = max(current_numbers, default=0) + 1
    return os.path.join(outputs_dir, f"{next_number:05d}{extension}")

def open_outputs_folder():
    outputs_dir = os.path.abspath("outputs")
    if os.name == 'nt':
        os.startfile(outputs_dir)
    elif os.name == 'posix':
        subprocess.Popen(["xdg-open", outputs_dir])
    else:
        print("[CMD] Opening folder not supported on this OS.")
    return f"Opened {outputs_dir}"

def load_wan_pipeline(model_choice, torch_dtype_str, num_persistent, lora_path=None, lora_alpha=None):
    print(f"[CMD] Loading model: {model_choice} with torch dtype: {torch_dtype_str} and num_persistent_param_in_dit: {num_persistent}")
    device = "cuda"
    torch_dtype = torch.float8_e4m3fn if torch_dtype_str == "torch.float8_e4m3fn" else torch.bfloat16

    model_manager = ModelManager(device="cpu")
    if model_choice == "1.3B":
        t5_path = get_common_file(os.path.join("models", "models_t5_umt5-xxl-enc-bf16.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-T2V-1.3B", "models_t5_umt5-xxl-enc-bf16.pth"))
        vae_path = get_common_file(os.path.join("models", "Wan2.1_VAE.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth"))
        model_manager.load_models(
            [
                os.path.join("models", "Wan-AI", "Wan2.1-T2V-1.3B", "diffusion_pytorch_model.safetensors"),
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )
    elif model_choice == "14B_text":
        t5_path = get_common_file(os.path.join("models", "models_t5_umt5-xxl-enc-bf16.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "models_t5_umt5-xxl-enc-bf16.pth"))
        vae_path = get_common_file(os.path.join("models", "Wan2.1_VAE.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "Wan2.1_VAE.pth"))
        model_manager.load_models(
            [
                [
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00001-of-00006.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00002-of-00006.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00003-of-00006.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00004-of-00006.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00005-of-00006.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00006-of-00006.safetensors")
                ],
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )
    elif model_choice == "14B_image_720p":
        clip_path = get_common_file(os.path.join("models", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"))
        t5_path = get_common_file(os.path.join("models", "models_t5_umt5-xxl-enc-bf16.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "models_t5_umt5-xxl-enc-bf16.pth"))
        vae_path = get_common_file(os.path.join("models", "Wan2.1_VAE.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "Wan2.1_VAE.pth"))
        model_manager.load_models(
            [
                [
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00001-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00002-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00003-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00004-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00005-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00006-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00007-of-00007.safetensors"),
                ],
                clip_path,
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )
    elif model_choice == "14B_image_480p":
        clip_path = get_common_file(os.path.join("models", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"))
        t5_path = get_common_file(os.path.join("models", "models_t5_umt5-xxl-enc-bf16.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "models_t5_umt5-xxl-enc-bf16.pth"))
        vae_path = get_common_file(os.path.join("models", "Wan2.1_VAE.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "Wan2.1_VAE.pth"))
        model_manager.load_models(
            [
                [
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00001-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00002-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00003-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00004-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00005-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00006-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00007-of-00007.safetensors"),
                ],
                clip_path,
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )
    else:
        raise ValueError("Invalid model choice")
    
    if lora_path is not None:
        print(f"[CMD] Loading LoRA from {lora_path} with alpha {lora_alpha}")
        model_manager.load_lora(lora_path, lora_alpha=lora_alpha)

    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)

    if str(num_persistent).strip().lower() == "none":
        num_persistent_val = None
    else:
        try:
            num_persistent_val = int(num_persistent)
        except Exception as e:
            print("[CMD] Warning: could not parse num_persistent_param_in_dit value, defaulting to 6000000000")
            num_persistent_val = 6000000000
    print(f"num_persistent_val {num_persistent_val}")
    pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_val)
    print("[CMD] Model loaded successfully.")
    return pipe

def get_lora_choices():
    lora_folder = "LoRAs"
    if not os.path.exists(lora_folder):
        os.makedirs(lora_folder)
        print("[CMD] 'LoRAs' folder not found. Created 'LoRAs' folder. Please add your LoRA .safetensors files.")
    files = [f for f in os.listdir(lora_folder) if f.endswith(".safetensors")]
    choices = ["None"] + sorted(files)
    return choices

def refresh_lora_list():
    return gr.update(choices=get_lora_choices(), value="None")

# ------------------------------
# Gradio Interface
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"])
    parser.add_argument("--prompt_extend_model", type=str, default=None)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    loaded_pipeline = None
    loaded_pipeline_config = {}
    cancel_flag = False
    prompt_expander = None

    css = """
    body {background: #1a1a1a; color: #fff; font-family: 'Arial', sans-serif;}
    .gradio-container {max-width: 1200px; margin: auto;}
    .button {background: #00aaff; border: none; border-radius: 5px; padding: 10px 20px; font-weight: bold;}
    .button:hover {background: #0077cc;}
    .input-section {background: #2a2a2a; border: 2px solid #00ccff; border-radius: 8px; padding: 20px; margin-bottom: 20px;}
    .header {text-align: center; color: #00aaff; margin-bottom: 20px;}
    """

    with gr.Blocks() as demo:
        gr.Markdown("GetGoingFast.pro WAN 2.1 I2V - T2V | [Listen to good music](https://music.youtube.com/playlist?list=OLAK5uy_kfreUHBgq8oX4pC3uYKBZVayEa0f941DQ)")
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Tab("Main Settings"):
                    with gr.Group(elem_classes="input-section"):
                        gr.Markdown("### Model & Input")
                        model_choice_radio = gr.Radio(
                            choices=[
                                "WAN 2.1 1.3B Text-to-Video",
                                "WAN 2.1 14B Text-to-Video",
                                "WAN 2.1 14B Image-to-Video 480P"
                            ],
                            label="Model Choice",
                            value=config_loaded.get("model_choice", "WAN 2.1 1.3B Text-to-Video")
                        )
                        prompt_box = gr.Textbox(label="Prompt", placeholder="Describe the video you want to generate", lines=5)
                        with gr.Row():
                            tar_lang = gr.Radio(choices=["CH", "EN"], label="Target language for prompt enhance", value=config_loaded.get("tar_lang", "EN"))
                            enhance_button = gr.Button("Prompt Enhance")
                        negative_prompt = gr.Textbox(label="Negative Prompt", value=config_loaded.get("negative_prompt", "still and motionless picture, static"), lines=2)
                        image_input = gr.Image(type="pil", label="Input Image (for image-to-video)", height=512)
                        generate_button = gr.Button("Generate", variant="primary")

                    with gr.Group(elem_classes="input-section"):
                        gr.Markdown("### Resolution & Output")
                        aspect_ratio_radio = gr.Radio(
                            choices=list(ASPECT_RATIOS_1_3b.keys()),
                            label="Aspect Ratio",
                            value=config_loaded.get("aspect_ratio", "16:9")
                        )
                        with gr.Row():
                            width_slider = gr.Slider(minimum=320, maximum=1536, step=16, value=config_loaded.get("width", 832), label="Width")
                            height_slider = gr.Slider(minimum=320, maximum=1536, step=16, value=config_loaded.get("height", 480), label="Height")
                            auto_crop_checkbox = gr.Checkbox(label="Auto Crop", value=config_loaded.get("auto_crop", True))
                        with gr.Row():
                            quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=config_loaded.get("quality", 5), label="Quality")
                            fps_slider = gr.Slider(minimum=8, maximum=30, step=1, value=config_loaded.get("fps", 16), label="FPS (for saving video)")
                            num_frames_slider = gr.Slider(minimum=1, maximum=300, step=1, value=config_loaded.get("num_frames", 81), label="Number of Frames")
                        with gr.Row():
                            save_prompt_checkbox = gr.Checkbox(label="Save prompt to file", value=config_loaded.get("save_prompt", True))
                            multiline_checkbox = gr.Checkbox(label="Multi-line prompt (each line is separate)", value=config_loaded.get("multiline", False))
                        num_generations = gr.Number(label="Number of Generations", value=config_loaded.get("num_generations", 1), precision=0)
                        with gr.Row():
                            use_random_seed_checkbox = gr.Checkbox(label="Use Random Seed", value=config_loaded.get("use_random_seed", True))
                            seed_input = gr.Textbox(label="Seed (if not using random)", placeholder="Enter seed", value=config_loaded.get("seed", ""))

                with gr.Tab("Advanced Settings"):
                    with gr.Group(elem_classes="input-section"):
                        gr.Markdown("### Advanced Configuration")
                        vram_preset_radio = gr.Radio(
                            choices=["4GB", "6GB", "8GB", "10GB", "12GB", "16GB", "24GB", "32GB", "48GB", "80GB"],
                            label="GPU VRAM Preset",
                            value=config_loaded.get("vram_preset", "24GB")
                        )
                        torch_dtype_radio = gr.Radio(
                            choices=["torch.float8_e4m3fn", "torch.bfloat16"],
                            label="Torch DType: float8 (FP8) reduces VRAM and RAM Usage",
                            value=config_loaded.get("torch_dtype", "torch.bfloat16")
                        )
                        tiled_checkbox = gr.Checkbox(label="Tiled VAE Decode (Disable for 1.3B model for 12GB or more GPUs)", value=config_loaded.get("tiled", True))
                        inference_steps_slider = gr.Slider(minimum=1, maximum=100, step=1, value=config_loaded.get("inference_steps", 50), label="Inference Steps")
                        with gr.Row():
                            cfg_scale_slider = gr.Slider(minimum=3, maximum=12, step=0.1, value=config_loaded.get("cfg_scale", 6.0), label="CFG Scale")
                            sigma_shift_slider = gr.Slider(minimum=3, maximum=12, step=0.1, value=config_loaded.get("sigma_shift", 6.0), label="Sigma Shift")
                        gr.Markdown("### Increase Video FPS with Practical-RIFE")
                        with gr.Row():
                            pr_rife_checkbox = gr.Checkbox(label="Apply Practical-RIFE", value=config_loaded.get("pr_rife", True))
                            pr_rife_radio = gr.Radio(choices=["2x FPS", "4x FPS"], label="FPS Multiplier", value=config_loaded.get("pr_rife_multiplier", "2x FPS"))
                        with gr.Row():
                            lora_dropdown = gr.Dropdown(
                                label="LoRA Model (Place .safetensors files in 'LoRAs' folder)",
                                choices=get_lora_choices(),
                                value=config_loaded.get("lora_model", "None")
                            )
                            lora_alpha_slider = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=config_loaded.get("lora_alpha", 1.0), label="LoRA Scale")
                            refresh_lora_button = gr.Button("Refresh LoRAs")

                with gr.Row():
                    cancel_button = gr.Button("Cancel")
                    open_outputs_button = gr.Button("Open Outputs Folder")

        enhance_button.click(fn=prompt_enc, inputs=[prompt_box, tar_lang], outputs=prompt_box)
        generate_button.click(
            fn=generate_videos,
            inputs=[
                prompt_box, tar_lang, negative_prompt, image_input,
                num_generations, save_prompt_checkbox, multiline_checkbox, use_random_seed_checkbox, seed_input,
                quality_slider, fps_slider,
                model_choice_radio, vram_preset_radio, num_persistent_text, torch_dtype_radio,
                num_frames_slider,
                aspect_ratio_radio, width_slider, height_slider, auto_crop_checkbox, tiled_checkbox, inference_steps_slider,
                pr_rife_checkbox, pr_rife_radio, cfg_scale_slider, sigma_shift_slider,
                lora_dropdown, lora_alpha_slider
            ],
            outputs=[video_output, last_seed_output]
        )
        cancel_button.click(fn=cancel_generation, outputs=[])
        open_outputs_button.click(fn=open_outputs_folder, outputs=[])
        model_choice_radio.change(
            fn=update_model_settings,
            inputs=[model_choice_radio, vram_preset_radio],
            outputs=[aspect_ratio_radio, width_slider, height_slider, num_persistent_text]
        )
        aspect_ratio_radio.change(
            fn=update_width_height,
            inputs=[aspect_ratio_radio, model_choice_radio],
            outputs=[width_slider, height_slider]
        )
        vram_preset_radio.change(
            fn=update_vram_on_change,
            inputs=[vram_preset_radio, model_choice_radio],
            outputs=num_persistent_text
        )
        refresh_lora_button.click(fn=refresh_lora_list, inputs=[], outputs=lora_dropdown)
        save_config_button.click(
            fn=save_config,
            inputs=[config_name_textbox, model_choice_radio, vram_preset_radio, aspect_ratio_radio, width_slider, height_slider,
                    auto_crop_checkbox, tiled_checkbox, inference_steps_slider, pr_rife_checkbox, pr_rife_radio, cfg_scale_slider, sigma_shift_slider,
                    num_persistent_text, torch_dtype_radio, lora_dropdown, lora_alpha_slider, negative_prompt, save_prompt_checkbox, multiline_checkbox,
                    num_generations, use_random_seed_checkbox, seed_input, quality_slider, fps_slider, num_frames_slider, tar_lang],
            outputs=[config_status, config_dropdown]
        )
        load_config_button.click(
            fn=load_config,
            inputs=[config_dropdown],
            outputs=[
                config_status, 
                model_choice_radio, vram_preset_radio, aspect_ratio_radio, width_slider, height_slider,
                auto_crop_checkbox, tiled_checkbox, inference_steps_slider, pr_rife_checkbox, pr_rife_radio, cfg_scale_slider, sigma_shift_slider,
                num_persistent_text, torch_dtype_radio, lora_dropdown, lora_alpha_slider, negative_prompt, save_prompt_checkbox, multiline_checkbox,
                num_generations, use_random_seed_checkbox, seed_input, quality_slider, fps_slider, num_frames_slider, tar_lang,
                config_name_textbox
            ]
        )

        demo.launch(share=args.share, inbrowser=True)
