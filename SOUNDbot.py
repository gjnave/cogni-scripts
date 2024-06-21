import subprocess
import os

url2 = "http://localhost:5001"
browser_path = r"venv\Scripts\midori\private_browsing.exe"


#Kobold config
command = [
    "koboldcpp.exe",
    "--model",
    "L3_8b_Stheno.gguf",
    "--sdmodel",
    "hybridRealityRealistic_hybridrealityV10.safetensors",
    "--sdquant",
    "--gpulayers",
    "99",
    "--preloadstory",
    "saved_story.json",
    "--smartcontext",
    "--quiet",
    "--highpriority",
    "--usecublas",
    "--contextsize",
    "2048",
    "--mmproj",
    "mmproj-model-f16.gguf",
    "--onready",
    f"{browser_path} {url2}"
]

# Execute the command
try:
    subprocess.run(command, check=True)
    print("Command executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error executing command: {e}")
