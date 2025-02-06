import os
import sys
import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel
import subprocess
import tempfile
import soundfile as sf
import librosa
from pathlib import Path
import platform
import time

class YuEInterface:
    def __init__(self):
        # Debug print for environment
        print(f"Platform: {platform.system()}")
        print(f"Current working directory: {os.getcwd()}")

        # Convert Windows-style paths to WSL paths if needed
        if "microsoft-standard-WSL" in platform.release():
            print("Running in WSL environment")
            self.base_path = Path("/mnt/d/staging/yue/yuewsl/yue")  # Adjusted based on your path
        else:
            self.base_path = Path(os.getcwd()) / "yue"

        print(f"Base path: {self.base_path}")

        self.inference_path = self.base_path / "inference"
        self.output_dir = self.base_path / "output"
        print(f"Output directory: {self.output_dir}")
        self.prompt_dir = self.base_path / "prompt_egs"

        # Create directories if they don't exist
        self._validate_directory_structure()

    def _validate_directory_structure(self):
        """Validate and create necessary directories"""
        print("Validating directory structure...")
        required_dirs = [self.base_path, self.inference_path, self.output_dir, self.prompt_dir]

        for directory in required_dirs:
            print(f"Checking directory: {directory}")
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    print(f"Created directory: {directory}")
                except Exception as e:
                    print(f"Error creating directory {directory}: {str(e)}")
                    raise RuntimeError(f"Failed to create directory {directory}: {str(e)}")

        # Verify inference script exists
        infer_script = self.inference_path / "infer.py"
        print(f"Checking for inference script at: {infer_script}")
        if not infer_script.exists():
            raise FileNotFoundError(f"Inference script not found at {infer_script}")
        print("Directory structure validation complete")

    def _get_latest_generated_file(self):
        """Helper method to get the latest generated audio file"""
        print(f"Looking for .mp3 files in: {self.output_dir}")
        
        # List all files in the output directory
        all_files = os.listdir(self.output_dir)
        
        # Filter for .mp3 files
        mp3_files = [f for f in all_files if f.endswith(".mp3")]
        print(f"Found .mp3 files: {mp3_files}")  # Debug: Print all found .mp3 files
        
        if not mp3_files:
            print("No generated .mp3 files found in output directory")
            return None
        
        # Find the latest file based on modification time
        latest_file = max(mp3_files, key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)))
        latest_file_path = os.path.join(self.output_dir, latest_file)
        print(f"Latest generated file: {latest_file_path}")
        
        return latest_file_path

    def _prepare_audio_for_gradio(self, audio_path):
        """Helper method to prepare audio file for Gradio interface"""
        try:
            if audio_path is None or not os.path.exists(audio_path):
                print(f"Audio file not found at: {audio_path}")
                return None, "No output file found"

            # Load the audio file using librosa to ensure it's valid
            try:
                audio_data, samplerate = librosa.load(audio_path, sr=None)
                print(f"Successfully loaded audio file: {audio_path}")
            except Exception as e:
                print(f"Error loading audio file: {str(e)}")
                return None, f"Error loading audio file: {str(e)}"

            # Return the path and a success message
            # Gradio will handle the file reading when given a valid path
            return str(audio_path), "Generation successful"
        except Exception as e:
            print(f"Error preparing audio: {str(e)}")
            return None, f"Error preparing audio: {str(e)}"
        

    def save_text_files(self, genre, lyrics):
        """Save genre and lyrics to text files"""
        try:
            genre_path = self.prompt_dir / "genre.txt"
            lyrics_path = self.prompt_dir / "lyrics.txt"

            print(f"Saving genre to: {genre_path}")
            print(f"Saving lyrics to: {lyrics_path}")

            genre_path.write_text(genre)
            lyrics_path.write_text(lyrics)

            return str(genre_path), str(lyrics_path)
        except Exception as e:
            print(f"Error in save_text_files: {str(e)}")
            raise RuntimeError(f"Failed to save text files: {str(e)}")

    def generate_music_cot(self, genre, lyrics, n_segments, batch_size):
        """Generate music using Chain of Thought mode"""
        try:
            print("Starting CoT generation...")
            genre_path, lyrics_path = self.save_text_files(genre, lyrics)

            # Convert paths to strings and ensure they use forward slashes
            inference_script = str(self.inference_path / "infer.py").replace('\\', '/')
            output_dir = str(self.output_dir).replace('\\', '/')
            genre_path = str(genre_path).replace('\\', '/')
            lyrics_path = str(lyrics_path).replace('\\', '/')

            print(f"Using inference script at: {inference_script}")
            print(f"Output directory: {output_dir}")

            command = [
                "python3" if platform.system() == "Linux" else sys.executable,
                inference_script,
                "--cuda_idx", "0",
                "--stage1_model", "m-a-p/YuE-s1-7B-anneal-en-cot",
                "--stage2_model", "m-a-p/YuE-s2-1B-general",
                "--genre_txt", genre_path,
                "--lyrics_txt", lyrics_path,
                "--run_n_segments", str(n_segments),
                "--stage2_batch_size", str(batch_size),
                "--output_dir", output_dir,
                "--max_new_tokens", "3000",
                "--repetition_penalty", "1.1"
            ]

            print(f"Running command: {' '.join(command)}")

            process = subprocess.Popen(
                command,
                cwd=str(self.inference_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy()
            )

            stdout, stderr = process.communicate()
            print(f"Process output: {stdout}")
            if stderr:
                print(f"Process errors: {stderr}")

            if process.returncode != 0:
                return None, f"Error during generation: {stderr}"

            # Wait a brief moment to ensure file writing is complete
            import time
            time.sleep(2)

            # Get the latest generated file
            latest_file = self._get_latest_generated_file()
            if latest_file is None:
                return None, "No output file generated"

            # Prepare the audio for Gradio
            return self._prepare_audio_for_gradio(str(latest_file))

        except Exception as e:
            print(f"Error in generate_music_cot: {str(e)}")
            return None, f"Error: {str(e)}"

    def generate_music_icl(self, genre, lyrics, n_segments, batch_size,
                           use_dual_tracks=False, vocal_track=None, instrumental_track=None,
                           single_track=None, start_time=0, end_time=30):
        """Generate music using In-Context Learning mode"""
        try:
            print("Starting ICL generation...")
            genre_path, lyrics_path = self.save_text_files(genre, lyrics)

            # Convert paths to strings and ensure they use forward slashes
            inference_script = str(self.inference_path / "infer.py").replace('\\', '/')
            output_dir = str(self.output_dir).replace('\\', '/')
            genre_path = str(genre_path).replace('\\', '/')
            lyrics_path = str(lyrics_path).replace('\\', '/')

            command = [
                "python3" if platform.system() == "Linux" else sys.executable,
                inference_script,
                "--cuda_idx", "0",
                "--stage1_model", "m-a-p/YuE-s1-7B-anneal-en-icl",
                "--stage2_model", "m-a-p/YuE-s2-1B-general",
                "--genre_txt", genre_path,
                "--lyrics_txt", lyrics_path,
                "--run_n_segments", str(n_segments),
                "--stage2_batch_size", str(batch_size),
                "--output_dir", output_dir,
                "--max_new_tokens", "3000",
                "--repetition_penalty", "1.1"
            ]

            if use_dual_tracks and vocal_track and instrumental_track:
                # Dual-track mode
                vocal_path = str(vocal_track).replace('\\', '/')
                instrumental_path = str(instrumental_track).replace('\\', '/')
                command.extend([
                    "--use_dual_tracks_prompt",
                    "--vocal_track_prompt_path", vocal_path,
                    "--instrumental_track_prompt_path", instrumental_path,
                    "--prompt_start_time", str(start_time),
                    "--prompt_end_time", str(end_time)
                ])
            elif single_track:
                # Single-track mode
                single_track_path = str(single_track).replace('\\', '/')
                command.extend([
                    "--use_audio_prompt",
                    "--audio_prompt_path", single_track_path,
                    "--prompt_start_time", str(start_time),
                    "--prompt_end_time", str(end_time)
                ])

            print(f"Running command: {' '.join(command)}")

            process = subprocess.Popen(
                command,
                cwd=str(self.inference_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy()
            )

            stdout, stderr = process.communicate()
            print(f"Process output: {stdout}")
            if stderr:
                print(f"Process errors: {stderr}")

            if process.returncode != 0:
                return None, f"Error during generation: {stderr}"

            # Wait a brief moment to ensure file writing is complete
            import time
            time.sleep(2)

            # Get the latest generated file
            latest_file = self._get_latest_generated_file()
            if latest_file is None:
                return None, "No output file generated"

            # Prepare the audio for Gradio
            return self._prepare_audio_for_gradio(str(latest_file))

        except Exception as e:
            print(f"Error in generate_music_icl: {str(e)}")
            return None, f"Error: {str(e)}"
            
def check_generated_files(output_dir):
    """
    Check for generated .mp3 files in the output directory and print their details.
    """
    # Convert output_dir to a Path object
    output_dir = Path(output_dir)
    
    # Check if the directory exists
    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}")
        return
    
    # List all .mp3 files in the directory
    mp3_files = list(output_dir.glob("*.mp3"))
    
    if not mp3_files:
        print(f"No .mp3 files found in: {output_dir}")
        return
    
    print(f"Found {len(mp3_files)} .mp3 files in: {output_dir}")
    print("=" * 50)
    
    # Print details for each file
    for file in mp3_files:
        file_stat = file.stat()
        print(f"File: {file.name}")
        print(f"  Path: {file}")
        print(f"  Size: {file_stat.st_size / 1024:.2f} KB")
        print(f"  Last Modified: {time.ctime(file_stat.st_mtime)}")
        print("-" * 50)
        
        


def create_interface():
    try:
        yue = YuEInterface()

        with gr.Blocks() as demo:
            gr.Markdown("# YuE Music Generation Interface")

            with gr.Tabs():
                # Chain of Thought (CoT) Tab
                with gr.Tab("Normal Generation (CoT)"):
                    with gr.Column():
                        genre_input = gr.Textbox(label="Genre", placeholder="Enter genre description...")
                        lyrics_input = gr.Textbox(label="Lyrics", placeholder="Enter lyrics...", lines=5)
                        n_segments = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Number of Segments")
                        batch_size = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Batch Size")
                        generate_btn = gr.Button("Generate Music")
                        output_audio = gr.Audio(label="Generated Music")
                        status_text = gr.Textbox(label="Status", interactive=False)

                    generate_btn.click(
                        fn=yue.generate_music_cot,
                        inputs=[genre_input, lyrics_input, n_segments, batch_size],
                        outputs=[output_audio, status_text]
                    )

                # In-Context Learning (ICL) Tab
                with gr.Tab("In-Context Learning (ICL)"):
                    with gr.Column():
                        icl_genre_input = gr.Textbox(label="Genre", placeholder="Enter genre description...")
                        icl_lyrics_input = gr.Textbox(label="Lyrics", placeholder="Enter lyrics...", lines=5)
                        icl_n_segments = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Number of Segments")
                        icl_batch_size = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Batch Size")

                        # Dual-track vs Single-track toggle
                        use_dual_tracks = gr.Checkbox(label="Use Dual Tracks", value=False)

                        # Dual-track inputs (visible only if use_dual_tracks is True)
                        with gr.Column() as dual_track_inputs:  # Removed visible=False
                            vocal_track = gr.Audio(label="Vocal Track (30s)", type="filepath")
                            instrumental_track = gr.Audio(label="Instrumental Track (30s)", type="filepath")

                        # Single-track inputs (visible only if use_dual_tracks is False)
                        with gr.Column() as single_track_inputs:  # Removed visible=False
                            single_track = gr.Audio(label="Single Track (30s)", type="filepath")

                        # Start and end time inputs
                        start_time = gr.Number(label="Start Time (seconds)", value=0)
                        end_time = gr.Number(label="End Time (seconds)", value=30)

                        # Generate button
                        icl_generate_btn = gr.Button("Generate Music")
                        icl_output_audio = gr.Audio(label="Generated Music")
                        icl_status_text = gr.Textbox(label="Status", interactive=False)

                    # Function to toggle between dual-track and single-track inputs
                    def toggle_track_inputs(use_dual):
                        return {
                            dual_track_inputs: use_dual,
                            single_track_inputs: not use_dual
                        }

                    # Attach the toggle function to the checkbox
                    use_dual_tracks.change(
                        fn=toggle_track_inputs,
                        inputs=[use_dual_tracks],
                        outputs=[dual_track_inputs, single_track_inputs]
                    )

                    # Wrapper function for ICL generation
                    def generate_icl_wrapper(*args):
                        use_dual = args[4]  # Whether to use dual tracks
                        if use_dual:
                            # Dual-track mode
                            return yue.generate_music_icl(
                                genre=args[0], lyrics=args[1], n_segments=args[2], batch_size=args[3],
                                use_dual_tracks=True, vocal_track=args[5], instrumental_track=args[6],
                                start_time=args[7], end_time=args[8]
                            )
                        else:
                            # Single-track mode
                            return yue.generate_music_icl(
                                genre=args[0], lyrics=args[1], n_segments=args[2], batch_size=args[3],
                                single_track=args[5], start_time=args[7], end_time=args[8]
                            )

                    # Attach the generate function to the button
                    icl_generate_btn.click(
                        fn=generate_icl_wrapper,
                        inputs=[
                            icl_genre_input, icl_lyrics_input, icl_n_segments, icl_batch_size,
                            use_dual_tracks, vocal_track, instrumental_track,
                            start_time, end_time
                        ],
                        outputs=[icl_output_audio, icl_status_text]
                    )
        return demo

    except Exception as e:
        print(f"Error creating interface: {str(e)}")
        return None

if __name__ == "__main__":
    
    
    
    print(f"*********************************")

    # Set the output directory path
    output_directory = "/mnt/d/staging/yue/yuewsl/yue/output"

    # Run the check
    check_generated_files(output_directory)
    demo = create_interface()
    if demo is not None:
        demo.launch(debug=True)
    else:
        print("Failed to create interface")
