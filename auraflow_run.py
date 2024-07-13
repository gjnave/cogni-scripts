import gradio as gr
import numpy as np
import random
from diffusers import AuraFlowPipeline
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

#torch.set_float32_matmul_precision("high")

#torch._inductor.config.conv_1x1_as_mm = True
#torch._inductor.config.coordinate_descent_tuning = True
#torch._inductor.config.epilogue_fusion = False
#torch._inductor.config.coordinate_descent_check_all_directions = True

pipe = AuraFlowPipeline.from_pretrained(
	"fal/AuraFlow",
    torch_dtype=torch.float16
).to("cuda")

#pipe.transformer.to(memory_format=torch.channels_last)
#pipe.vae.to(memory_format=torch.channels_last)

#pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
#pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

def infer(prompt, negative_prompt="", seed=42, randomize_seed=False, width=1024, height=1024, guidance_scale=5.0, num_inference_steps=28, progress=gr.Progress(track_tqdm=True)):

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
        
    generator = torch.Generator().manual_seed(seed)
    
    image = pipe(
        prompt = prompt, 
        negative_prompt = negative_prompt,
        width=width,
        height=height,
        guidance_scale = guidance_scale, 
        num_inference_steps = num_inference_steps, 
        generator = generator
    ).images[0] 
    
    return image, seed

examples = [
    "A photo of a lavender cat",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

if torch.cuda.is_available():
    power_device = "GPU"
else:
    power_device = "CPU"

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
        # AuraFlow 0.1
        Demo of the [AuraFlow 0.1](https://huggingface.co/fal/AuraFlow) 6.8B parameters open source diffusion transformer model
        [[blog](https://blog.fal.ai/auraflow/)] [[model](https://huggingface.co/fal/AuraFlow)] [[fal](https://fal.ai/models/fal-ai/aura-flow)]
        """)
        
        with gr.Row():
            
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            
            run_button = gr.Button("Run", scale=0)
        
        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):
            
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
            )
            
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            
            with gr.Row():
                
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
                
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
            
            with gr.Row():
                
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=5.0,
                )
                
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=28,
                )
        
        gr.Examples(
            examples = examples,
            fn = infer,
            inputs = [prompt],
            outputs = [result, seed],
            cache_examples="lazy"
        )

    gr.on(
        triggers=[run_button.click, prompt.submit, negative_prompt.submit],
        fn = infer,
        inputs = [prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs = [result, seed]
    )

demo.queue().launch()