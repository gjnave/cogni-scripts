"""
Qwen3-TTS Gradio Web UI
A clean, intuitive interface for all Qwen3-TTS capabilities
"""
import os
import gradio as gr
import torch
import soundfile as sf
from pathlib import Path
import tempfile

# Setup SoX for Windows
try:
    import static_sox
    static_sox.add_paths(weak=True)
except:
    pass

from qwen_tts import Qwen3TTSModel

# Global model cache
MODEL_CACHE = {}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Default model paths
MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_PATHS = {
    "CustomVoice": str(MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    "VoiceDesign": str(MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    "Base": str(MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-Base"),
}

# Supported speakers for CustomVoice model
SPEAKERS = {
    "Vivian (Chinese, Young Female)": "Vivian",
    "Serena (Chinese, Warm Female)": "Serena",
    "Uncle Fu (Chinese, Mature Male)": "Uncle_Fu",
    "Dylan (Chinese Beijing, Young Male)": "Dylan",
    "Eric (Chinese Sichuan, Lively Male)": "Eric",
    "Ryan (English, Dynamic Male)": "Ryan",
    "Aiden (English, Clear Male)": "Aiden",
    "Ono Anna (Japanese, Playful Female)": "Ono_Anna",
    "Sohee (Korean, Warm Female)": "Sohee",
}

LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]


def load_model(model_type, model_path=None):
    """Load and cache model"""
    if model_path is None or not model_path.strip():
        model_path = DEFAULT_PATHS.get(model_type)
    
    # If local path doesn't exist, try to use it as HF repo ID
    if not Path(model_path).exists():
        # Use HF repo IDs as fallback
        hf_repos = {
            "CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "Base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        }
        model_path = hf_repos.get(model_type, model_path)
    
    cache_key = f"{model_type}:{model_path}"
    
    if cache_key not in MODEL_CACHE:
        try:
            MODEL_CACHE[cache_key] = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=DEVICE,
                dtype=DTYPE,
                attn_implementation="sdpa",  # Use SDPA (works everywhere)
            )
        except Exception as e:
            raise gr.Error(f"Failed to load model: {str(e)}")
    
    return MODEL_CACHE[cache_key]


def generate_custom_voice(text, language, speaker, instruction, model_path, progress=gr.Progress()):
    """Generate audio using CustomVoice model"""
    if not text.strip():
        raise gr.Error("Please enter some text to synthesize")
    
    progress(0.2, desc="Loading model...")
    model = load_model("CustomVoice", model_path if model_path.strip() else None)
    
    progress(0.5, desc="Generating audio...")
    speaker_id = SPEAKERS.get(speaker, speaker)
    
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language if language != "Auto" else None,
        speaker=speaker_id,
        instruct=instruction if instruction.strip() else None,
    )
    
    progress(1.0, desc="Done!")
    
    # Save to temp file
    output_path = tempfile.mktemp(suffix=".wav")
    sf.write(output_path, wavs[0], sr)
    
    return output_path


def generate_voice_design(text, language, voice_description, model_path, progress=gr.Progress()):
    """Generate audio using VoiceDesign model"""
    if not text.strip():
        raise gr.Error("Please enter some text to synthesize")
    
    if not voice_description.strip():
        raise gr.Error("Please provide a voice description")
    
    progress(0.2, desc="Loading model...")
    model = load_model("VoiceDesign", model_path if model_path.strip() else None)
    
    progress(0.5, desc="Generating audio...")
    
    wavs, sr = model.generate_voice_design(
        text=text,
        language=language if language != "Auto" else None,
        instruct=voice_description,
    )
    
    progress(1.0, desc="Done!")
    
    output_path = tempfile.mktemp(suffix=".wav")
    sf.write(output_path, wavs[0], sr)
    
    return output_path


def generate_voice_clone(text, language, ref_audio, ref_text, use_x_vector, model_path, progress=gr.Progress()):
    """Generate audio using voice cloning"""
    if not text.strip():
        raise gr.Error("Please enter some text to synthesize")
    
    if ref_audio is None:
        raise gr.Error("Please provide a reference audio file")
    
    # Auto-enable x_vector mode if no ref_text
    if not ref_text.strip() and not use_x_vector:
        use_x_vector = True
        gr.Info("No reference text provided - using x_vector mode (slightly lower quality)")
    
    progress(0.2, desc="Loading model...")
    model = load_model("Base", model_path if model_path.strip() else None)
    
    progress(0.5, desc="Generating audio...")
    
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language if language != "Auto" else None,
        ref_audio=ref_audio,
        ref_text=ref_text if ref_text.strip() else None,
        x_vector_only_mode=use_x_vector,
    )
    
    progress(1.0, desc="Done!")
    
    output_path = tempfile.mktemp(suffix=".wav")
    sf.write(output_path, wavs[0], sr)
    
    return output_path


# Build Gradio UI
with gr.Blocks(title="Qwen3-TTS", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🎙️ Qwen3-TTS | [Get Going Fast](https://getgoingfast.pro)")
    gr.Markdown("Professional text-to-speech with custom voices, voice design, and voice cloning")
    gr.Markdown("**[DJ Grizzly Edition](https://www.youtube.com/@dj__grizzly)**")
    
    with gr.Tabs() as tabs:
        
        # ===== CUSTOM VOICE TAB =====
        with gr.Tab("🎤 Custom Voice", id="custom"):
            gr.Markdown("### Use Pre-defined Premium Voices")
            gr.Markdown("Choose from 9 professional voice presets covering different languages, genders, and ages")
            
            with gr.Row():
                with gr.Column(scale=2):
                    cv_text = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter the text you want to convert to speech...",
                        lines=5,
                    )
                    
                    with gr.Row():
                        cv_language = gr.Dropdown(
                            choices=LANGUAGES,
                            value="Auto",
                            label="Language",
                            info="Auto-detect or select specific language",
                        )
                        cv_speaker = gr.Dropdown(
                            choices=list(SPEAKERS.keys()),
                            value=list(SPEAKERS.keys())[0],
                            label="Voice",
                            info="Choose a voice preset",
                        )
                    
                    cv_instruction = gr.Textbox(
                        label="Style Instruction (Optional)",
                        placeholder="e.g., 'speak slowly and warmly' or 'excited and energetic'",
                        lines=2,
                    )
                    
                    cv_model_path = gr.Textbox(
                        label="Model Path (Optional)",
                        placeholder="Leave empty to use default CustomVoice model",
                        value="",
                    )
                    
                    cv_generate = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    cv_output = gr.Audio(label="Generated Audio", type="filepath")
            
            # Examples
            gr.Examples(
                examples=[
                    ["Hello! Welcome to Qwen3-TTS. This is an amazing text-to-speech system.", "English", "Ryan (English, Dynamic Male)", ""],
                    ["I've noticed something interesting - I'm really good at reading people's emotions.", "English", "Vivian (Chinese, Young Female)", "speak in a very angry tone"],
                    ["Good morning! How are you today?", "Japanese", "Ono Anna (Japanese, Playful Female)", ""],
                ],
                inputs=[cv_text, cv_language, cv_speaker, cv_instruction],
            )
        
        # ===== VOICE DESIGN TAB =====
        with gr.Tab("🎨 Voice Design", id="design"):
            gr.Markdown("### Create Custom Voices from Text Descriptions")
            gr.Markdown("Describe the voice you want and the AI will create it for you")
            
            with gr.Row():
                with gr.Column(scale=2):
                    vd_text = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter the text you want to convert to speech...",
                        lines=5,
                    )
                    
                    vd_language = gr.Dropdown(
                        choices=LANGUAGES,
                        value="Auto",
                        label="Language",
                    )
                    
                    vd_description = gr.Textbox(
                        label="Voice Description",
                        placeholder="Describe the voice characteristics, e.g., 'Young male, confident, deep voice with slight rasp' or 'Sweet teenage girl, gentle and warm, slightly high pitched'",
                        lines=4,
                    )
                    
                    vd_model_path = gr.Textbox(
                        label="Model Path (Optional)",
                        placeholder="Leave empty to use default VoiceDesign model",
                        value="",
                    )
                    
                    vd_generate = gr.Button("🎨 Design & Generate", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    vd_output = gr.Audio(label="Generated Audio", type="filepath")
            
            # Examples
            gr.Examples(
                examples=[
                    ["What? No! I mean yes but not like... I just think you're really smart!", "English", "Male, 17 years old, tenor range, nervous but gaining confidence"],
                    ["Hey big brother, you're back! I've been waiting for you for so long, give me a hug!", "English", "Young playful female voice, high-pitched with obvious ups and downs, creating a clingy, cutesy effect"],
                ],
                inputs=[vd_text, vd_language, vd_description],
            )
        
        # ===== VOICE CLONE TAB =====
        with gr.Tab("🎭 Voice Clone", id="clone"):
            gr.Markdown("### Clone Any Voice from a Sample")
            gr.Markdown("Upload a 3+ second audio clip and clone the voice to say anything")
            
            with gr.Row():
                with gr.Column(scale=2):
                    vc_text = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter what you want the cloned voice to say...",
                        lines=5,
                    )
                    
                    vc_language = gr.Dropdown(
                        choices=LANGUAGES,
                        value="Auto",
                        label="Language",
                    )
                    
                    vc_ref_audio = gr.Audio(
                        label="Reference Audio (3+ seconds recommended)",
                        type="filepath",
                    )
                    
                    vc_ref_text = gr.Textbox(
                        label="Reference Transcript (Recommended for better quality)",
                        placeholder="Type exactly what is said in the reference audio...",
                        lines=3,
                        info="Providing the transcript improves cloning quality. Leave empty to use x_vector mode.",
                    )
                    
                    vc_x_vector = gr.Checkbox(
                        label="Force x_vector mode (ignore transcript)",
                        value=False,
                        info="Uses only voice embedding, not transcript. Slightly lower quality.",
                    )
                    
                    vc_model_path = gr.Textbox(
                        label="Model Path (Optional)",
                        placeholder="Leave empty to use default Base model",
                        value="",
                    )
                    
                    vc_generate = gr.Button("🎭 Clone & Generate", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    vc_output = gr.Audio(label="Generated Audio", type="filepath")
            
            gr.Markdown("**Tip:** For best results, use a clean 3-10 second audio clip with minimal background noise")
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("Powered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Built with Gradio")
    
    # Wire up buttons
    cv_generate.click(
        fn=generate_custom_voice,
        inputs=[cv_text, cv_language, cv_speaker, cv_instruction, cv_model_path],
        outputs=cv_output,
    )
    
    vd_generate.click(
        fn=generate_voice_design,
        inputs=[vd_text, vd_language, vd_description, vd_model_path],
        outputs=vd_output,
    )
    
    vc_generate.click(
        fn=generate_voice_clone,
        inputs=[vc_text, vc_language, vc_ref_audio, vc_ref_text, vc_x_vector, vc_model_path],
        outputs=vc_output,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7864,
        share=False,
    )
