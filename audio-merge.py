import subprocess
import os

def select_video(prompt):
    """Function to prompt user to select a video file using tkinter dialog."""
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    
    root = Tk()
    root.withdraw()  # Hide the main window

    file_path = askopenfilename(title=prompt, filetypes=[("Video files", "*.mp4;*.webm")])

    root.destroy()  # Close the tkinter window

    return file_path

def main():
    # Step 1: Select video to extract audio from
    video_to_extract = select_video("Select video to extract audio from")
    if not video_to_extract:
        print("No video selected. Exiting.")
        return
    
    # Step 2: Select video to add audio to
    video_to_add_audio = select_video("Select video to add audio to")
    if not video_to_add_audio:
        print("No video selected. Exiting.")
        return
    
    # Step 3: Extract audio from video_to_extract to a temporary WAV file
    temp_audio_file = "temp_audio.wav"
    extract_cmd = ["ffmpeg", "-i", video_to_extract, "-vn", "-c:a", "pcm_s16le", temp_audio_file]
    subprocess.run(extract_cmd, check=True)

    # Step 4: Add extracted audio to video_to_add_audio and save as output.mp4
    output_file = "output.mp4"
    add_audio_cmd = ["ffmpeg", "-i", video_to_add_audio, "-i", temp_audio_file, "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", output_file]
    subprocess.run(add_audio_cmd, check=True)

    # Step 5: Clean up temporary audio file
    if os.path.exists(temp_audio_file):
        os.remove(temp_audio_file)

    print(f"Audio extracted from {video_to_extract} and added to {video_to_add_audio}. Output saved as {output_file}.")

if __name__ == "__main__":
    main()
