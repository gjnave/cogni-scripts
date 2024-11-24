import subprocess
import os
import gradio as gr
import os
from gradio_magicquill import MagicQuill
import random
import torch
import numpy as np
from PIL import Image, ImageOps
import base64
import io
from fastapi import FastAPI, Request
import uvicorn
import requests
from MagicQuill import folder_paths
from MagicQuill.llava_new import LLaVAModel
from MagicQuill.scribble_color_edit import ScribbleColorEditModel
from datetime import datetime

llavaModel = LLaVAModel()
scribbleColorEditModel = ScribbleColorEditModel()

url = "http://localhost:7860"

def tensor_to_base64(tensor):
    tensor = tensor.squeeze(0) * 255.
    pil_image = Image.fromarray(tensor.cpu().byte().numpy())
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str

def read_base64_image(base64_image):
    if base64_image.startswith("data:image/png;base64,"):
        base64_image = base64_image.split(",")[1]
    elif base64_image.startswith("data:image/jpeg;base64,"):
        base64_image = base64_image.split(",")[1]
    elif base64_image.startswith("data:image/webp;base64,"):
        base64_image = base64_image.split(",")[1]
    else:
        raise ValueError("Unsupported image format.")
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))
    image = ImageOps.exif_transpose(image)
    return image

def create_alpha_mask(base64_image):
    """Create an alpha mask from the alpha channel of an image."""
    image = read_base64_image(base64_image)
    mask = torch.zeros((1, image.height, image.width), dtype=torch.float32, device="cpu")
    if 'A' in image.getbands():
        alpha_channel = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask[0] = 1.0 - torch.from_numpy(alpha_channel)
    return mask

def load_and_preprocess_image(base64_image, convert_to='RGB', has_alpha=False):
    """Load and preprocess a base64 image."""
    image = read_base64_image(base64_image)
    image = image.convert(convert_to)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)[None,]
    return image_tensor

def load_and_resize_image(base64_image, convert_to='RGB', max_size=512):
    """Load and preprocess a base64 image, resize if necessary."""
    image = read_base64_image(base64_image)
    image = image.convert(convert_to)
    width, height = image.size
    scaling_factor = max_size / min(width, height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    image = image.resize(new_size, Image.LANCZOS)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)[None,]
    return image_tensor

def prepare_images_and_masks(total_mask, original_image, add_color_image, add_edge_image, remove_edge_image):
    total_mask = create_alpha_mask(total_mask)
    original_image_tensor = load_and_preprocess_image(original_image)
    if add_color_image:
        add_color_image_tensor = load_and_preprocess_image(add_color_image)
    else:
        add_color_image_tensor = original_image_tensor
    
    add_edge_mask = create_alpha_mask(add_edge_image) if add_edge_image else torch.zeros_like(total_mask)
    remove_edge_mask = create_alpha_mask(remove_edge_image) if remove_edge_image else torch.zeros_like(total_mask)
    return add_color_image_tensor, original_image_tensor, total_mask, add_edge_mask, remove_edge_mask


def guess(original_image_tensor, add_color_image_tensor, add_edge_mask):
    description, ans1, ans2 = llavaModel.process(original_image_tensor, add_color_image_tensor, add_edge_mask)
    ans_list = []
    if ans1 and ans1 != "":
        ans_list.append(ans1)
    if ans2 and ans2 != "":
        ans_list.append(ans2)

    return ", ".join(ans_list)

def guess_prompt_handler(original_image, add_color_image, add_edge_image):
    original_image_tensor = load_and_preprocess_image(original_image)
    
    if add_color_image:
        add_color_image_tensor = load_and_preprocess_image(add_color_image)
    else:
        add_color_image_tensor = original_image_tensor
    
    width, height = original_image_tensor.shape[1], original_image_tensor.shape[2]
    add_edge_mask = create_alpha_mask(add_edge_image) if add_edge_image else torch.zeros((1, height, width), dtype=torch.float32, device="cpu")
    res = guess(original_image_tensor, add_color_image_tensor, add_edge_mask)
    return res
    
def save_generated_image(image_base64):
    """Save the generated image to disk with timestamp."""
    try:
        # Remove data URL prefix if present
        if image_base64.startswith("data:image/png;base64,"):
            image_base64 = image_base64.split(",")[1]
            
        # Decode base64 to image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_image_{timestamp}.png"
        
        # Create output directory if it doesn't exist
        output_dir = "generated_images"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save image
        filepath = os.path.join(output_dir, filename)
        image.save(filepath, "PNG")
        return filepath
    except Exception as e:
        print(f"Error saving image: {e}")
        return None


def generate(ckpt_name, total_mask, original_image, add_color_image, add_edge_image, remove_edge_image, positive_prompt, negative_prompt, grow_size, stroke_as_edge, fine_edge, edge_strength, color_strength, inpaint_strength, seed, steps, cfg, sampler_name, scheduler):
    add_color_image, original_image, total_mask, add_edge_mask, remove_edge_mask = prepare_images_and_masks(total_mask, original_image, add_color_image, add_edge_image, remove_edge_image)
    progress = None
    if torch.sum(remove_edge_mask).item() > 0 and torch.sum(add_edge_mask).item() == 0:
        if positive_prompt == "":
            positive_prompt = "empty scene"
        edge_strength /= 3.

    latent_samples, final_image, lineart_output, color_output = scribbleColorEditModel.process(
        ckpt_name,
        original_image, 
        add_color_image, 
        positive_prompt, 
        negative_prompt, 
        total_mask, 
        add_edge_mask, 
        remove_edge_mask, 
        grow_size, 
        stroke_as_edge, 
        fine_edge,
        edge_strength, 
        color_strength,  
        inpaint_strength, 
        seed, 
        steps, 
        cfg, 
        sampler_name, 
        scheduler,
        progress
    )

    final_image_base64 = tensor_to_base64(final_image)
    return final_image_base64

def generate_image_handler(x, ckpt_name, negative_prompt, fine_edge, grow_size, edge_strength, color_strength, inpaint_strength, seed, steps, cfg, sampler_name, scheduler):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    ms_data = x['from_frontend']
    positive_prompt = x['from_backend']['prompt']
    payload = {
        "ckpt_name": ckpt_name,
        "total_mask": ms_data['total_mask'],
        "original_image": ms_data['original_image'],
        "add_color_image": ms_data['add_color_image'],
        "add_edge_image": ms_data['add_edge_image'],
        "remove_edge_image": ms_data['remove_edge_image'],
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "grow_size": grow_size,
        "stroke_as_edge": "enable",
        "fine_edge": fine_edge,
        "edge_strength": edge_strength,
        "color_strength": color_strength,
        "inpaint_strength": inpaint_strength,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler_name,
        "scheduler": scheduler
    }
    res = requests.post(f"{url}/magic_quill/generate_image", json=payload).json()
    if 'error' in res:
        print(res['error'])
        x["from_backend"]["generated_image"] = None
    x["from_backend"]["generated_image"] = res['res']
    return x

css = '''
.row {
    width: 90%;
    margin: auto;
}
footer {
    visibility: 
    hidden
}
'''

with gr.Blocks(css=css) as demo:
    with gr.Row(elem_classes="row"):
        ms = MagicQuill()
    with gr.Row(elem_classes="row"):
        with gr.Column():
            gen_btn = gr.Button("Generate", variant="primary")
            save_btn = gr.Button("Save Image", variant="secondary")
            save_status = gr.Textbox(label="Save Status", interactive=False)
        with gr.Column():
            with gr.Accordion("parameters", open=False):
                default_model = "SD1.5\\realisticVisionV60B1_v51VAE.safetensors"
                ckpt_name = gr.Dropdown(
                    label="Base Model Name",
                    choices=folder_paths.get_filename_list("checkpoints"),
                    value='SD1.5\\realisticVisionV60B1_v51VAE.safetensors',
                    interactive=True
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="",
                    interactive=True
                )
                # stroke_as_edge = gr.Radio(
                #     label="Stroke as Edge",
                #     choices=['enable', 'disable'],
                #     value='enable',
                #     interactive=True
                # )
                fine_edge = gr.Radio(
                    label="Fine Edge",
                    choices=['enable', 'disable'],
                    value='disable',
                    interactive=True
                )
                grow_size = gr.Slider(
                    label="Grow Size",
                    minimum=0,
                    maximum=100,
                    value=15,
                    step=1,
                    interactive=True
                )
                edge_strength = gr.Slider(
                    label="Edge Strength",
                    minimum=0.0,
                    maximum=5.0,
                    value=0.55,
                    step=0.01,
                    interactive=True
                )
                color_strength = gr.Slider(
                    label="Color Strength",
                    minimum=0.0,
                    maximum=5.0,
                    value=0.55,
                    step=0.01,
                    interactive=True
                )
                inpaint_strength = gr.Slider(
                    label="Inpaint Strength",
                    minimum=0.0,
                    maximum=5.0,
                    value=1.0,
                    step=0.01,
                    interactive=True
                )
                seed = gr.Number(
                    label="Seed",
                    value=-1,
                    precision=0,
                    interactive=True
                )
                steps = gr.Slider(
                    label="Steps",
                    minimum=1,
                    maximum=50,
                    value=20,
                    interactive=True
                )
                cfg = gr.Slider(
                    label="CFG",
                    minimum=0.0,
                    maximum=100.0,
                    value=5.0,
                    step=0.1,
                    interactive=True
                )
                sampler_name = gr.Dropdown(
                    label="Sampler Name",
                    choices=["euler", "euler_ancestral", "heun", "heunpp2","dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2"],
                    value='euler_ancestral',
                    interactive=True
                )
                scheduler = gr.Dropdown(
                    label="Scheduler",
                    choices=["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"],
                    value='karras',
                    interactive=True
                )

    def save_current_image(ms):
        if not ms or not ms.get('from_backend', {}).get('generated_image'):
            return "No image to save"
        
        saved_path = save_generated_image(ms['from_backend']['generated_image'])
        if saved_path:
            return f"Image saved successfully at: {saved_path}"
        return "Failed to save image"

    gen_btn.click(
        generate_image_handler,
        inputs=[ms, ckpt_name, negative_prompt, fine_edge, grow_size, edge_strength, 
               color_strength, inpaint_strength, seed, steps, cfg, sampler_name, scheduler],
        outputs=ms
    )
    
    save_btn.click(
        save_current_image,
        inputs=[ms],
        outputs=save_status
    )

app = FastAPI()

@app.post("/magic_quill/generate_image")
async def generate_image(request: Request):
    data = await request.json()
    try :
        res = generate(
            data['ckpt_name'],
            data['total_mask'],
            data['original_image'],
            data['add_color_image'],
            data['add_edge_image'],
            data['remove_edge_image'],
            data['positive_prompt'],
            data['negative_prompt'],
            data['grow_size'],
            data['stroke_as_edge'],
            data['fine_edge'],
            data['edge_strength'],
            data['color_strength'],
            data['inpaint_strength'],
            data['seed'],
            data['steps'],
            data['cfg'],
            data['sampler_name'],
            data['scheduler']
        )
        return {'res': res}
    except Exception as e:
        print(e)
        return{'error': str(e)}

@app.post("/magic_quill/guess_prompt")
async def guess_prompt(request: Request):
    data = await request.json()
    res = guess_prompt_handler(data['original_image'], data['add_color_image'], data['add_edge_image'])
    return res

@app.post("/magic_quill/process_background_img")
async def process_background_img(request: Request):
    img = await request.json()
    resized_img_tensor = load_and_resize_image(img)
    resized_img_base64 = "data:image/png;base64," + tensor_to_base64(resized_img_tensor)
    # add more processing here
    return resized_img_base64

app = gr.mount_gradio_app(app, demo, "/")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
    # demo.launch()
