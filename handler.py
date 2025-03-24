# handler.py
import os
import shutil
import time
import json
import base64
import requests
from clearvoice import ClearVoice

def enhance_audio(input_path, output_path, model_pipeline, temp_dir='temp'):
    """
    Enhance audio using a pipeline of ClearVoice models.
    
    Args:
        input_path (str): Path to the input audio file
        output_path (str): Path where the final enhanced audio will be saved
        model_pipeline (list of dict): List of dictionaries containing 'task' and 'model_name' for each step
        temp_dir (str): Directory to store intermediate files
    """
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Track the current file being processed
    current_file = input_path
    
    # Iterate through the model pipeline
    for i, model_info in enumerate(model_pipeline):
        task = model_info['task']
        model_name = model_info['model_name']
        
        print(f"Step {i+1}: {task} using {model_name}")
        
        # Generate a unique intermediate file name
        intermediate_path = os.path.join(temp_dir, f'intermediate_{i}_{model_name}.wav')
        
        # Process the audio with the current model
        process_audio(task, model_name, current_file, intermediate_path)
        
        # Update the current file to the intermediate file
        current_file = intermediate_path
    
    # Move the final result to the output path
    shutil.move(current_file, output_path)
    
    print(f"Final enhanced audio saved to {output_path}")
    return output_path

def process_audio(task, model_name, input_path, output_path):
    """
    Process audio using a specific ClearVoice model.
    
    Args:
        task (str): The task to perform ('speech_enhancement', 'speech_super_resolution', etc.)
        model_name (str): The name of the model to use
        input_path (str): Path to the input audio file
        output_path (str): Path where the processed audio will be saved
    """
    cv = ClearVoice(task=task, model_names=[model_name])
    processed_wav = cv(input_path=input_path, online_write=False)
    cv.write(processed_wav, output_path=output_path)
    return output_path

def handler(job):
    """
    RunPod serverless handler function with direct file input/output
    """
    job_input = job["input"]
    
    # Make directories for inputs, outputs, and temp files
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    # Get timestamp for unique filename
    timestamp = str(time.time()).replace(".", "")
    input_path = f"inputs/input_{timestamp}.wav"
    output_path = f"outputs/output_{timestamp}.wav"
    
    # Handle the input audio file
    if "file" in job["input"]:
        # In RunPod, the "file" key would contain the path to the uploaded file
        input_path = job["input"]["file"]
    elif "input_url" in job["input"]:
        # Download from URL
        response = requests.get(job["input"]["input_url"])
        with open(input_path, "wb") as f:
            f.write(response.content)
    elif "input_data" in job["input"]:
        # Decode base64 data
        audio_data = base64.b64decode(job["input"]["input_data"])
        with open(input_path, "wb") as f:
            f.write(audio_data)
    else:
        return {"error": "No input audio provided. Please provide 'file', 'input_url', or 'input_data'"}
    
    # Get model pipeline from input or use default
    model_pipeline = job_input.get("model_pipeline", [
        {'task': 'speech_enhancement', 'model_name': 'MossFormer2_SE_48K'},
        {'task': 'speech_super_resolution', 'model_name': 'MossFormer2_SR_48K'},
        {'task': 'speech_enhancement', 'model_name': 'MossFormer2_SE_48K'}
    ])
    
    try:
        # Process the audio
        enhance_audio(input_path, output_path, model_pipeline, temp_dir='temp')
        
        # Read the output file into base64 for the response
        with open(output_path, "rb") as f:
            output_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Return both the file path and base64 data
        return {
            "output": {
                "file_path": output_path,  # Path to the output file on the server
                "audio_data": output_data,  # Base64 encoded audio for direct download
                "models_used": [model["model_name"] for model in model_pipeline]
            }
        }
    
    except Exception as e:
        # If any error occurs, return it
        return {"error": str(e)}