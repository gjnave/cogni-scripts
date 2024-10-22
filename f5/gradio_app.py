# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file
import yt_dlp
import os
import re
import tempfile
from pathlib import Path
import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from pydub import AudioSegment
import subprocess

from model import DiT, UNetT
from model.utils import (
    save_spectrogram,
)
from model.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)

vocos = load_vocoder()


# load models
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(
    DiT, F5TTS_model_cfg, str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
)

E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
E2TTS_ema_model = load_model(
    UNetT, E2TTS_model_cfg, str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
)

#****************************************
def get_speaker_files():
    speaker_dir = Path("./speakers")
    speaker_files = [f.stem for f in speaker_dir.glob("*.wav")]
    return {f: str(speaker_dir / f"{f}.wav") for f in speaker_files}

def update_reference_audio(speaker):
    return speaker_files[speaker]
    
def infer(ref_audio, ref_text, gen_text, model_choice, remove_silence, cross_fade_duration):
    sample_rate = 22050
    duration = 5  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Generate spectrogram
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='hz', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title('Spectrogram')
    
    # Save spectrogram to a file
    spectrogram_path = "spectrogram.png"
    plt.savefig(spectrogram_path)
    plt.close(fig)
    
    return (sample_rate, audio), spectrogram_path

def update_speed(speed):
    # Your existing update_speed function implementation
    pass
    
speaker_files = get_speaker_files()

# ***************************************
def timestamp_to_seconds(timestamp):
    """Convert timestamp to seconds"""
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    try:
        parts = timestamp.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        else:
            return float(timestamp)
    except (ValueError, AttributeError):
        return 0

def download_youtube_audio(url, progress=None):
    clips_dir = Path("clips")
    clips_dir.mkdir(exist_ok=True)
    
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(clips_dir / 'audio'),  # Remove .mp3 extension here
            'quiet': False,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            final_path = str(clips_dir / 'audio.mp3')
            return final_path, None, info.get('title', 'Unknown Title')
            
    except Exception as e:
        return None, str(e), None

def trim_audio(input_file, start_time, end_time):
    """Trim audio file using FFmpeg"""
    try:
        # Always use our known audio file
        input_file = str(Path("clips") / "audio.mp3")
        output_file = str(Path("clips") / "trimmed.mp3")
        
        # Convert timestamps to seconds
        start_seconds = timestamp_to_seconds(start_time) if start_time else 0
        if end_time:
            end_seconds = timestamp_to_seconds(end_time)
            duration = end_seconds - start_seconds
            cmd = ['ffmpeg', '-i', input_file, '-ss', str(start_seconds), '-t', str(duration), '-acodec', 'copy', output_file, '-y']
        else:
            cmd = ['ffmpeg', '-i', input_file, '-ss', str(start_seconds), '-acodec', 'copy', output_file, '-y']
            
        print(f"Running FFmpeg command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            return None, f"FFmpeg error: {process.stderr}"
            
        return output_file, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def on_trim_click(audio_file, start_time, end_time):
    if not audio_file:
        return {
            trim_status: "Please download an audio file first",
            trim_preview_row: gr.update(visible=False),
            trimmed_audio_preview: None
        }
    
    output_file, error = trim_audio(audio_file, start_time, end_time)
    
    if error:
        return {
            trim_status: f"Error: {error}",
            trim_preview_row: gr.update(visible=False),
            trimmed_audio_preview: None
        }
    else:
        return {
            trim_status: f"Successfully trimmed to: {output_file}",
            trim_preview_row: gr.update(visible=True),
            trimmed_audio_preview: output_file
        }
# Add this before the final app definition
with gr.Blocks() as app_youtube:
    gr.Markdown("""
    # YouTube Audio Downloader and Trimmer
    
    ## Step 1: Download YouTube Audio
    Enter a YouTube URL to download the audio in MP3 format.
    """)
    
    with gr.Row():
        url_input = gr.Textbox(
            label="YouTube URL", 
            placeholder="https://www.youtube.com/watch?v=..."
        )
    
    download_btn = gr.Button("Download Audio", variant="primary")
    status_output = gr.Textbox(label="Status", interactive=False)
    
    # Preview section
    with gr.Row(visible=False) as preview_row:
        audio_preview = gr.Audio(label="Preview")
        title_display = gr.Textbox(label="Video Title", interactive=False)
    
    # Trim section
    gr.Markdown("""
    ## Step 2: Trim Audio (Optional)
    After downloading, you can trim the audio to specific timestamps.
    
    Timestamp formats: HH:MM:SS, MM:SS, or seconds
    Example: 1:30 or 01:30 or 90 (all mean 1 minute 30 seconds)
    """)
    
    with gr.Row():
        trim_start = gr.Textbox(
            label="Start Time", 
            placeholder="0:00"
        )
        trim_end = gr.Textbox(
            label="End Time", 
            placeholder="1:30"
        )
    
    trim_btn = gr.Button("Trim Audio", variant="secondary")
    trim_status = gr.Textbox(label="Trim Status", interactive=False)
    
    with gr.Row(visible=False) as trim_preview_row:
        trimmed_audio_preview = gr.Audio(label="Trimmed Audio Preview")
    
    def on_download_click(url):
        if not url:
            return {
                status_output: "Please enter a URL",
                preview_row: gr.update(visible=False),
                audio_preview: None,
                title_display: ""
            }
        
        filepath, error, title = download_youtube_audio(url)
        
        if error:
            return {
                status_output: f"Error: {error}",
                preview_row: gr.update(visible=False),
                audio_preview: None,
                title_display: ""
            }
        else:
            return {
                status_output: f"Successfully downloaded to: {filepath}",
                preview_row: gr.update(visible=True),
                audio_preview: filepath,
                title_display: title
            }
    
    def on_trim_click(audio_file, start_time, end_time):
        if not audio_file:
            return {
                trim_status: "Please download an audio file first",
                trim_preview_row: gr.update(visible=False),
                trimmed_audio_preview: None
            }
            
        output_file, error = trim_audio(audio_file, start_time, end_time)
        
        if error:
            return {
                trim_status: f"Error: {error}",
                trim_preview_row: gr.update(visible=False),
                trimmed_audio_preview: None
            }
        else:
            return {
                trim_status: f"Successfully trimmed to: {output_file}",
                trim_preview_row: gr.update(visible=True),
                trimmed_audio_preview: output_file
            }
    
    download_btn.click(
        on_download_click,
        inputs=[url_input],
        outputs=[status_output, preview_row, audio_preview, title_display]
    )
    
    trim_btn.click(
        on_trim_click,
        inputs=[audio_preview, trim_start, trim_end],
        outputs=[trim_status, trim_preview_row, trimmed_audio_preview]
    )

#*********************************

def infer(ref_audio_orig, ref_text, gen_text, model, remove_silence, cross_fade_duration=0.15, speed=1):
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=gr.Info)

    if model == "F5-TTS":
        ema_model = F5TTS_ema_model
    elif model == "E2-TTS":
        ema_model = E2TTS_ema_model

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        cross_fade_duration=cross_fade_duration,
        speed=speed,
        show_info=gr.Info,
        progress=gr.Progress(),
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path


def generate_podcast(
    script, speaker1_name, ref_audio1, ref_text1, speaker2_name, ref_audio2, ref_text2, model, remove_silence
):
    # Split the script into speaker blocks
    speaker_pattern = re.compile(f"^({re.escape(speaker1_name)}|{re.escape(speaker2_name)}):", re.MULTILINE)
    speaker_blocks = speaker_pattern.split(script)[1:]  # Skip the first empty element

    generated_audio_segments = []

    for i in range(0, len(speaker_blocks), 2):
        speaker = speaker_blocks[i]
        text = speaker_blocks[i + 1].strip()

        # Determine which speaker is talking
        if speaker == speaker1_name:
            ref_audio = ref_audio1
            ref_text = ref_text1
        elif speaker == speaker2_name:
            ref_audio = ref_audio2
            ref_text = ref_text2
        else:
            continue  # Skip if the speaker is neither speaker1 nor speaker2

        # Generate audio for this block
        audio, _ = infer(ref_audio, ref_text, text, model, remove_silence)

        # Convert the generated audio to a numpy array
        sr, audio_data = audio

        # Save the audio data as a WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sr)
            audio_segment = AudioSegment.from_wav(temp_file.name)

        generated_audio_segments.append(audio_segment)

        # Add a short pause between speakers
        pause = AudioSegment.silent(duration=500)  # 500ms pause
        generated_audio_segments.append(pause)

    # Concatenate all audio segments
    final_podcast = sum(generated_audio_segments)

    # Export the final podcast
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        podcast_path = temp_file.name
        final_podcast.export(podcast_path, format="wav")

    return podcast_path


def parse_speechtypes_text(gen_text):
    # Pattern to find (Emotion)
    pattern = r"\((.*?)\)"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_emotion = "Regular"

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                segments.append({"emotion": current_emotion, "text": text})
        else:
            # This is emotion
            emotion = tokens[i].strip()
            current_emotion = emotion

    return segments


with gr.Blocks() as app_credits:
    gr.Markdown("""
# Credits

* [mrfakename](https://github.com/fakerybakery) for the original [online demo](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [RootingInLoad](https://github.com/RootingInLoad) for the podcast generation
* [jpgallegoar](https://github.com/jpgallegoar) for multiple speech-type generation
* [Cognibuild](https://www.cognibuild.ai) to a MUCH lesser level for the mere downloader and speaker pack integration
""")
with gr.Blocks() as app_tts:
    gr.Markdown("# Batched TTS")
    
    speaker_dropdown = gr.Dropdown(
        choices=list(speaker_files.keys()),
        label="Choose Speaker",
        type="value"
    )
    ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
    gen_text_input = gr.Textbox(label="Text to Generate", lines=10)
    model_choice = gr.Radio(
        choices=["F5-TTS", "E2-TTS"], label="Choose TTS Model", value="F5-TTS"
    )
    generate_btn = gr.Button("Synthesize", variant="primary")
    
    with gr.Accordion("Advanced Settings", open=False):
        ref_text_input = gr.Textbox(
            label="Reference Text",
            info="Leave blank to automatically transcribe the reference audio. If you enter text it will override automatic transcription.",
            lines=2,
        )
        remove_silence = gr.Checkbox(
            label="Remove Silences",
            info="The model tends to produce silences, especially on longer audio. We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. This will also increase generation time.",
            value=False,
        )
        speed_slider = gr.Slider(
            label="Speed",
            minimum=0.3,
            maximum=2.0,
            value=1.0,  # Assuming default speed is 1.0
            step=0.1,
            info="Adjust the speed of the audio.",
        )
        cross_fade_duration_slider = gr.Slider(
            label="Cross-Fade Duration (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
            info="Set the duration of the cross-fade between audio clips.",
        )
    
    speed_slider.change(update_speed, inputs=speed_slider)
    audio_output = gr.Audio(label="Synthesized Audio")
    spectrogram_output = gr.Image(label="Spectrogram")
    
    # Update reference audio when a speaker is selected
    speaker_dropdown.change(
        update_reference_audio,
        inputs=[speaker_dropdown],
        outputs=[ref_audio_input]
    )
    
    generate_btn.click(
        infer,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            model_choice,
            remove_silence,
            cross_fade_duration_slider,
        ],
        outputs=[audio_output, spectrogram_output],
    )

def podcast_generation(script, speaker1, ref_audio1, ref_text1, speaker2, ref_audio2, ref_text2, model, remove_silence):
    return generate_podcast(script, speaker1, ref_audio1, ref_text1, speaker2, ref_audio2, ref_text2, model, remove_silence)

    generate_podcast_btn.click(
        podcast_generation,
        inputs=[
            script_input,
            speaker1_name,
            ref_audio_input1,
            ref_text_input1,
            speaker2_name,
            ref_audio_input2,
            ref_text_input2,
            podcast_model_choice,
            podcast_remove_silence,
        ],
        outputs=podcast_output,
    )

with gr.Blocks() as app_podcast:
    gr.Markdown("# Podcast Generation")
    
    with gr.Row():
        with gr.Column():
            speaker1_name = gr.Textbox(label="Speaker 1 Name")
            speaker1_dropdown = gr.Dropdown(
                choices=list(speaker_files.keys()),
                label="Choose Speaker 1",
                type="value"
            )
            ref_audio_input1 = gr.Audio(label="Reference Audio (Speaker 1)", type="filepath")
            ref_text_input1 = gr.Textbox(label="Reference Text (Speaker 1)", lines=2)
        
        with gr.Column():
            speaker2_name = gr.Textbox(label="Speaker 2 Name")
            speaker2_dropdown = gr.Dropdown(
                choices=list(speaker_files.keys()),
                label="Choose Speaker 2",
                type="value"
            )
            ref_audio_input2 = gr.Audio(label="Reference Audio (Speaker 2)", type="filepath")
            ref_text_input2 = gr.Textbox(label="Reference Text (Speaker 2)", lines=2)
    
    script_input = gr.Textbox(
        label="Podcast Script", 
        lines=10, 
        placeholder="Enter the script with speaker names at the start of each block, e.g.:\nSean: How did you start studying...\n\nMeghan: I came to my interest in technology...\nIt was a long journey...\n\nSean: That's fascinating. Can you elaborate..."
    )
    
    podcast_model_choice = gr.Radio(
        choices=["F5-TTS", "E2-TTS"], label="Choose TTS Model", value="F5-TTS"
    )
    podcast_remove_silence = gr.Checkbox(
        label="Remove Silences",
        value=True,
    )
    generate_podcast_btn = gr.Button("Generate Podcast", variant="primary")
    podcast_output = gr.Audio(label="Generated Podcast")
    
    # Update reference audio when speakers are selected
    speaker1_dropdown.change(
        update_reference_audio,
        inputs=[speaker1_dropdown],
        outputs=[ref_audio_input1]
    )
    speaker2_dropdown.change(
        update_reference_audio,
        inputs=[speaker2_dropdown],
        outputs=[ref_audio_input2]
    )
    
    generate_podcast_btn.click(
        podcast_generation,
        inputs=[
            script_input,
            speaker1_name,
            ref_audio_input1,
            ref_text_input1,
            speaker2_name,
            ref_audio_input2,
            ref_text_input2,
            podcast_model_choice,
            podcast_remove_silence,
        ],
        outputs=podcast_output,
    )


def parse_emotional_text(gen_text):
    # Pattern to find (Emotion)
    pattern = r"\((.*?)\)"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_emotion = "Regular"

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                segments.append({"emotion": current_emotion, "text": text})
        else:
            # This is emotion
            emotion = tokens[i].strip()
            current_emotion = emotion

    return segments


with gr.Blocks() as app_emotional:
    # New section for emotional generation
    gr.Markdown(
        """
    # Multiple Speech-Type Generation

    This section allows you to upload different audio clips for each speech type. 'Regular' emotion is mandatory. You can add additional speech types by clicking the "Add Speech Type" button. Enter your text in the format shown below, and the system will generate speech using the appropriate emotions. If unspecified, the model will use the regular speech type. The current speech type will be used until the next speech type is specified.

    **Example Input:**

    (Regular) Hello, I'd like to order a sandwich please. (Surprised) What do you mean you're out of bread? (Sad) I really wanted a sandwich though... (Angry) You know what, darn you and your little shop, you suck! (Whisper) I'll just go back home and cry now. (Shouting) Why me?!
    """
    )

    gr.Markdown(
        "Upload different audio clips for each speech type. 'Regular' emotion is mandatory. You can add additional speech types by clicking the 'Add Speech Type' button."
    )

    # Regular speech type (mandatory)
    with gr.Row():
        regular_name = gr.Textbox(value="Regular", label="Speech Type Name", interactive=False)
        regular_audio = gr.Audio(label="Regular Reference Audio", type="filepath")
        regular_ref_text = gr.Textbox(label="Reference Text (Regular)", lines=2)

    # Additional speech types (up to 99 more)
    max_speech_types = 100
    speech_type_names = []
    speech_type_audios = []
    speech_type_ref_texts = []
    speech_type_delete_btns = []

    for i in range(max_speech_types - 1):
        with gr.Row():
            name_input = gr.Textbox(label="Speech Type Name", visible=False)
            audio_input = gr.Audio(label="Reference Audio", type="filepath", visible=False)
            ref_text_input = gr.Textbox(label="Reference Text", lines=2, visible=False)
            delete_btn = gr.Button("Delete", variant="secondary", visible=False)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)
        speech_type_delete_btns.append(delete_btn)

    # Button to add speech type
    add_speech_type_btn = gr.Button("Add Speech Type")

    # Keep track of current number of speech types
    speech_type_count = gr.State(value=0)

    # Function to add a speech type
    def add_speech_type_fn(speech_type_count):
        if speech_type_count < max_speech_types - 1:
            speech_type_count += 1
            # Prepare updates for the components
            name_updates = []
            audio_updates = []
            ref_text_updates = []
            delete_btn_updates = []
            for i in range(max_speech_types - 1):
                if i < speech_type_count:
                    name_updates.append(gr.update(visible=True))
                    audio_updates.append(gr.update(visible=True))
                    ref_text_updates.append(gr.update(visible=True))
                    delete_btn_updates.append(gr.update(visible=True))
                else:
                    name_updates.append(gr.update())
                    audio_updates.append(gr.update())
                    ref_text_updates.append(gr.update())
                    delete_btn_updates.append(gr.update())
        else:
            # Optionally, show a warning
            # gr.Warning("Maximum number of speech types reached.")
            name_updates = [gr.update() for _ in range(max_speech_types - 1)]
            audio_updates = [gr.update() for _ in range(max_speech_types - 1)]
            ref_text_updates = [gr.update() for _ in range(max_speech_types - 1)]
            delete_btn_updates = [gr.update() for _ in range(max_speech_types - 1)]
        return [speech_type_count] + name_updates + audio_updates + ref_text_updates + delete_btn_updates

    add_speech_type_btn.click(
        add_speech_type_fn,
        inputs=speech_type_count,
        outputs=[speech_type_count]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts
        + speech_type_delete_btns,
    )

    # Function to delete a speech type
    def make_delete_speech_type_fn(index):
        def delete_speech_type_fn(speech_type_count):
            # Prepare updates
            name_updates = []
            audio_updates = []
            ref_text_updates = []
            delete_btn_updates = []

            for i in range(max_speech_types - 1):
                if i == index:
                    name_updates.append(gr.update(visible=False, value=""))
                    audio_updates.append(gr.update(visible=False, value=None))
                    ref_text_updates.append(gr.update(visible=False, value=""))
                    delete_btn_updates.append(gr.update(visible=False))
                else:
                    name_updates.append(gr.update())
                    audio_updates.append(gr.update())
                    ref_text_updates.append(gr.update())
                    delete_btn_updates.append(gr.update())

            speech_type_count = max(0, speech_type_count - 1)

            return [speech_type_count] + name_updates + audio_updates + ref_text_updates + delete_btn_updates

        return delete_speech_type_fn

    for i, delete_btn in enumerate(speech_type_delete_btns):
        delete_fn = make_delete_speech_type_fn(i)
        delete_btn.click(
            delete_fn,
            inputs=speech_type_count,
            outputs=[speech_type_count]
            + speech_type_names
            + speech_type_audios
            + speech_type_ref_texts
            + speech_type_delete_btns,
        )

    # Text input for the prompt
    gen_text_input_emotional = gr.Textbox(label="Text to Generate", lines=10)

    # Model choice
    model_choice_emotional = gr.Radio(choices=["F5-TTS", "E2-TTS"], label="Choose TTS Model", value="F5-TTS")

    with gr.Accordion("Advanced Settings", open=False):
        remove_silence_emotional = gr.Checkbox(
            label="Remove Silences",
            value=True,
        )

    # Generate button
    generate_emotional_btn = gr.Button("Generate Emotional Speech", variant="primary")

    # Output audio
    audio_output_emotional = gr.Audio(label="Synthesized Audio")

    def generate_emotional_speech(
        regular_audio,
        regular_ref_text,
        gen_text,
        *args,
    ):
        num_additional_speech_types = max_speech_types - 1
        speech_type_names_list = args[:num_additional_speech_types]
        speech_type_audios_list = args[num_additional_speech_types : 2 * num_additional_speech_types]
        speech_type_ref_texts_list = args[2 * num_additional_speech_types : 3 * num_additional_speech_types]
        model_choice = args[3 * num_additional_speech_types]
        remove_silence = args[3 * num_additional_speech_types + 1]

        # Collect the speech types and their audios into a dict
        speech_types = {"Regular": {"audio": regular_audio, "ref_text": regular_ref_text}}

        for name_input, audio_input, ref_text_input in zip(
            speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
        ):
            if name_input and audio_input:
                speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input}

        # Parse the gen_text into segments
        segments = parse_speechtypes_text(gen_text)

        # For each segment, generate speech
        generated_audio_segments = []
        current_emotion = "Regular"

        for segment in segments:
            emotion = segment["emotion"]
            text = segment["text"]

            if emotion in speech_types:
                current_emotion = emotion
            else:
                # If emotion not available, default to Regular
                current_emotion = "Regular"

            ref_audio = speech_types[current_emotion]["audio"]
            ref_text = speech_types[current_emotion].get("ref_text", "")

            # Generate speech for this segment
            audio, _ = infer(ref_audio, ref_text, text, model_choice, remove_silence, 0)
            sr, audio_data = audio

            generated_audio_segments.append(audio_data)

        # Concatenate all audio segments
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            return (sr, final_audio_data)
        else:
            gr.Warning("No audio generated.")
            return None

    generate_emotional_btn.click(
        generate_emotional_speech,
        inputs=[
            regular_audio,
            regular_ref_text,
            gen_text_input_emotional,
        ]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts
        + [
            model_choice_emotional,
            remove_silence_emotional,
        ],
        outputs=audio_output_emotional,
    )

    # Validation function to disable Generate button if speech types are missing
    def validate_speech_types(gen_text, regular_name, *args):
        num_additional_speech_types = max_speech_types - 1
        speech_type_names_list = args[:num_additional_speech_types]

        # Collect the speech types names
        speech_types_available = set()
        if regular_name:
            speech_types_available.add(regular_name)
        for name_input in speech_type_names_list:
            if name_input:
                speech_types_available.add(name_input)

        # Parse the gen_text to get the speech types used
        segments = parse_emotional_text(gen_text)
        speech_types_in_text = set(segment["emotion"] for segment in segments)

        # Check if all speech types in text are available
        missing_speech_types = speech_types_in_text - speech_types_available

        if missing_speech_types:
            # Disable the generate button
            return gr.update(interactive=False)
        else:
            # Enable the generate button
            return gr.update(interactive=True)

    gen_text_input_emotional.change(
        validate_speech_types,
        inputs=[gen_text_input_emotional, regular_name] + speech_type_names,
        outputs=generate_emotional_btn,
    )
with gr.Blocks() as app:
    gr.Markdown(
        """
# E2/F5 TTS *PLUS* 
[Cognibuild Edition](www.patreon.com/cognibuild)
This edition has a Video Downloader w/ Trimmer, and 150 speaker pack
* [Listen to Good Music](https://open.spotify.com/artist/0oH5qisu13DpnT7DucnV9d)

This is a local web UI for F5 TTS with advanced batch processing support. This app supports the following TTS models:

* [F5-TTS](https://arxiv.org/abs/2410.06885) (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching)
* [E2 TTS](https://arxiv.org/abs/2406.18009) (Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS)

The checkpoints support English and Chinese.

If you're having issues, try converting your reference audio to WAV or MP3, clipping it to 15s, and shortening your prompt.

**NOTE: Reference text will be automatically transcribed with Whisper if not provided. For best results, keep your reference clips short (<15s). Ensure the audio is fully uploaded before generating.**
"""
    )
    gr.TabbedInterface([app_tts, app_podcast, app_emotional, app_youtube, app_credits], ["TTS", "Podcast", "Multi-Style", "YouTube", "Credits"])


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
def main(port, host, share, api):
    global app
    print("Starting app...")
    app.queue(api_open=api).launch(server_name=host, server_port=port, share=share, show_api=api)


if __name__ == "__main__":
    app.queue().launch()
