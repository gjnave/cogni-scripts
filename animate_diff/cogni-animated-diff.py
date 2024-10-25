import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import gradio as gr
from pathlib import Path
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def find_path(name: str, path: str = None) -> str:
    if path is None:
        path = os.getcwd()
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        logger.info(f"{name} found: {path_name}")
        return path_name
    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None
    return find_path(name, parent_directory)

def ensure_directory(path: Path) -> Path:
    """Create directory if it doesn't exist and verify it's writable"""
    path.mkdir(parents=True, exist_ok=True)
    
    # Verify directory exists and is writable
    if not path.exists():
        raise RuntimeError(f"Failed to create directory: {path}")
    if not os.access(path, os.W_OK):
        raise RuntimeError(f"Directory is not writable: {path}")
    
    return path
    
def get_available_checkpoints() -> list[str]:
    """Get list of available checkpoint models from the models/checkpoints folder."""
    folder_path = "models/checkpoints"
    return [filename for filename in os.listdir(folder_path) if filename.endswith(".safetensors")]

def get_available_loras() -> list[str]:
    """Get list of available checkpoint models from the models/checkpoints folder."""
    folder_path = "models/loras"
    return [filename for filename in os.listdir(folder_path) if filename.endswith(".safetensors")]
    
    
def generate_animation(
    prompt: str,
    negative_prompt: str,
    checkpoint: str,  # Added checkpoint parameter
    lora_model: str,
    width: int = 360,
    height: int = 568,
    num_frames: int = 100,
    steps: int = 81,
    cfg: float = 7,
    seed: int = None,
    fps: int = 8
) -> tuple[str, str]:
    """Generate an animated sequence using AnimateDiff"""
    
    if seed is None:
        seed = random.randint(1, 2**64)
    
    # Get ComfyUI base directory
    comfy_dir = Path(find_path("ComfyUI"))
    
    # Use ComfyUI's output directory structure
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = comfy_dir / "output"
    frame_path = output_dir / timestamp / "frames"
    gif_path = output_dir / timestamp / "animations"
    
    # Ensure directories exist
    ensure_directory(frame_path)
    ensure_directory(gif_path)
    
    logger.info(f"ComfyUI directory: {comfy_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Frames directory: {frame_path}")
    logger.info(f"GIF directory: {gif_path}")
    
    try:
        with torch.inference_mode():
            # Load VAE
            logger.info("Loading VAE...")
            vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
            vae = vaeloader.load_vae(
                vae_name="vaeFtMse840000EmaPruned_vaeFtMse840k.safetensors"
            )

            # Load checkpoint using the selected checkpoint from the dropdown
            logger.info(f"Loading checkpoint: {checkpoint}")  # Using the passed checkpoint parameter
            checkpointloader = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            checkpoint_model = checkpointloader.load_checkpoint(
                ckpt_name=checkpoint
            )
            
            # Encode prompts
            logger.info("Encoding prompts...")
            clip_encoder = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            positive_embed = clip_encoder.encode(
                text=f"(Masterpiece, best quality:1.2), {prompt}",
                clip=get_value_at_index(checkpoint_model, 1),
            )
            
            negative_embed = clip_encoder.encode(
                text=f"(bad quality, worst quality:1.2), {negative_prompt}",
                clip=get_value_at_index(checkpoint_model, 1),
            )

            # Setup latent
            logger.info("Setting up latent...")
            empty_latent = NODE_CLASS_MAPPINGS["ADE_EmptyLatentImageLarge"]()
            latent = empty_latent.generate(
                width=width, height=height, batch_size=num_frames
            )

            # Setup AnimateDiff options
            logger.info("Setting up AnimateDiff options...")
            context_options = NODE_CLASS_MAPPINGS["ADE_StandardUniformContextOptions"]()
            context_settings = context_options.create_options(
                context_length=16,
                context_stride=1,
                context_overlap=4,
                fuse_method="pyramid",
                use_on_equal_length=False,
                start_percent=0,
                guarantee_steps=1,
            )

            sampling_settings = NODE_CLASS_MAPPINGS["ADE_AnimateDiffSamplingSettings"]()
            sample_settings = sampling_settings.create_settings(
                batch_offset=0,
                noise_type="FreeNoise",
                seed_gen="comfy",
                seed_offset=0,
                adapt_denoise_steps=False,
            )

            # Load and apply motion model
            logger.info("Loading motion model...")
            motion_loader = NODE_CLASS_MAPPINGS["ADE_LoadAnimateDiffModel"]()
            motion_model = motion_loader.load_motion_model(
                model_name="mm_sd_v15_v2.ckpt"
            )

            motion_applier = NODE_CLASS_MAPPINGS["ADE_ApplyAnimateDiffModelSimple"]()
            applied_motion = motion_applier.apply_motion_model(
                motion_model=get_value_at_index(motion_model, 0)
            )

            # Setup sampling
            logger.info("Setting up sampling...")
            evolved_sampler = NODE_CLASS_MAPPINGS["ADE_UseEvolvedSampling"]()
            sampling = evolved_sampler.use_evolved_sampling(
                beta_schedule="autoselect",
                model=get_value_at_index(checkpoint_model, 0),
                m_models=get_value_at_index(applied_motion, 0),
                context_options=get_value_at_index(context_settings, 0),
                sample_settings=get_value_at_index(sample_settings, 0),
            )

            # Generate frames
            logger.info("Generating frames...")
            ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            samples = ksampler.sample(
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name="euler_ancestral",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(sampling, 0),
                positive=get_value_at_index(positive_embed, 0),
                negative=get_value_at_index(negative_embed, 0),
                latent_image=get_value_at_index(latent, 0),
            )

            # Decode frames
            logger.info("Decoding frames...")
            vae_decoder = NODE_CLASS_MAPPINGS["VAEDecode"]()
            decoded = vae_decoder.decode(
                samples=get_value_at_index(samples, 0),
                vae=get_value_at_index(vae, 0),
            )

            # Save individual frames
            logger.info("Saving frames...")
            frame_filename = f"frame_{timestamp}"
            saver = NODE_CLASS_MAPPINGS["SaveImage"]()
            frames = saver.save_images(
                filename_prefix=frame_filename,
                images=get_value_at_index(decoded, 0),
            )

            # Generate GIF
            logger.info("Creating animation...")
            gif_filename = f"animation_{timestamp}"
            video_combiner = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
            gif = video_combiner.combine_video(
                frame_rate=fps,
                loop_count=0,
                filename_prefix=gif_filename,
                format="image/gif",
                pingpong=False,
                save_output=True,
                images=get_value_at_index(decoded, 0),
            )
            
            # Wait for files to be written and verify their existence
            time.sleep(2)
            
            # Search for the generated files in ComfyUI's output directory
            frame_files = list(output_dir.rglob(f"{frame_filename}*.png"))
            gif_files = list(output_dir.rglob(f"{gif_filename}*.gif"))
            
            logger.info(f"Found {len(frame_files)} frame files")
            logger.info(f"Found {len(gif_files)} gif files")
            
            if not frame_files:
                raise RuntimeError(f"No frame files were generated in {output_dir}")
            if not gif_files:
                raise RuntimeError(f"No gif files were generated in {output_dir}")
                
            # Return paths as strings
            return str(frame_files[0]), str(gif_files[0])
            
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}", exc_info=True)
        raise gr.Error(f"Generation failed: {str(e)}")

def create_ui():
    app = gr.Blocks(title="AnimateDiff Generator")
    
    with app:
        gr.Markdown("# AnimateDiff Generator")
        gr.Markdown("Generate animated sequences using AnimateDiff")
        
        with gr.Row():
            with gr.Column():
                # Add checkpoint selection at the top
                checkpoint = gr.Dropdown(
                    label="Select Checkpoint Model",
                    choices=get_available_checkpoints(),
                    value="dreamshaper_8.safetensors",
                    type="value"
                )
                
                lora = gr.Dropdown(
                    label="Select LoRA Model - ie. For Green-Screen Add 'Green Background' to Prompt",  # Fixed label
                    choices=[""] + get_available_loras(),
                    value="",
                    type="value"
                )
                
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe what you want to generate...",
                    value="Wonder woman standing in front of a wall and smiling at the crowd"
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="malformed, deformed, elongated, multiple body parts"
                )
                
                with gr.Row():
                    width = gr.Slider(
                        minimum=64,
                        maximum=1024,
                        step=64,
                        value=360,
                        label="Width"
                    )
                    height = gr.Slider(
                        minimum=64,
                        maximum=1024,
                        step=64,
                        value=568,
                        label="Height"
                    )
                
                with gr.Row():
                    num_frames = gr.Slider(
                        minimum=8,
                        maximum=200,
                        step=1,
                        value=50,
                        label="Number of Frames"
                    )
                    fps = gr.Slider(
                        minimum=1,
                        maximum=30,
                        step=1,
                        value=8,
                        label="FPS"
                    )
                
                with gr.Row():
                    steps = gr.Slider(
                        minimum=20,
                        maximum=150,
                        step=1,
                        value=50,
                        label="Steps"
                    )
                    cfg = gr.Slider(
                        minimum=1,
                        maximum=20,
                        step=0.5,
                        value=7,
                        label="CFG Scale"
                    )
                
                seed = gr.Number(
                    label="Seed (leave empty for random)",
                    precision=0
                )
                
                generate_btn = gr.Button("Generate")
                status_text = gr.Markdown("Ready")
            
            with gr.Column():
                frame_output = gr.Image(label="Latest Frame")
                gif_output = gr.Video(label="Animation")
        
        # Move the click event binding inside the app context
        generate_btn.click(
            fn=on_generate,
            inputs=[
                checkpoint,
                lora,
                prompt,
                negative_prompt,
                width,
                height,
                num_frames,
                steps,
                cfg,
                seed,
                fps
            ],
            outputs=[frame_output, gif_output, status_text]
        )

    return app

def on_generate(ckpt, lora_model, prompt, neg_prompt, w, h, n_frames, step_count, cfg_scale, seed_val, fps_val):
    try:
        status_text = "Starting generation with LoRA..."  # Changed to string instead of accessing .value
        logger.info(f"Starting generation with checkpoint: {ckpt} and LoRA: {lora_model}")
        
        frame_path, gif_path = generate_animation(
            prompt=prompt,
            negative_prompt=neg_prompt,
            checkpoint=ckpt,
            lora_model=lora_model,  # Fixed parameter name to match generate_animation
            width=w,
            height=h,
            num_frames=n_frames,
            steps=step_count,
            cfg=cfg_scale,
            seed=seed_val,
            fps=fps_val
        )
        
        logger.info(f"Generation complete. Frame: {frame_path}, GIF: {gif_path}")
        return [frame_path, gif_path, "Generation complete!"]
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise gr.Error(error_msg)

        # Updated generate button click to include the lora_model input
        generate_btn.click(
            fn=on_generate,
            inputs=[
                checkpoint,
                lora,  # New input for LoRA
                prompt,
                negative_prompt,
                width,
                height,
                num_frames,
                steps,
                cfg,
                seed,
                fps
            ],
            outputs=[frame_output, gif_output, status_text]
        )

    return app

if __name__ == "__main__":
    # Import asyncio here to avoid issues with event loop
    import asyncio
    import gradio as gr
    
    # Add ComfyUI directory to system path
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        logger.info(f"'{comfyui_path}' added to sys.path")
    else:
        logger.error("ComfyUI directory not found!")
        sys.exit(1)

    # Import required ComfyUI modules
    import execution
    from nodes import init_extra_nodes, NODE_CLASS_MAPPINGS
    import server

    # Initialize ComfyUI server and nodes
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    init_extra_nodes()
    
    logger.info("Starting Gradio interface...")
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
