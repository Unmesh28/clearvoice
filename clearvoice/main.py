import os
import shutil
import base64
import runpod
from clearvoice import ClearVoice
import tempfile
import logging
import json
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def enhance_audio(input_path, output_path, model_pipeline, temp_dir='temp'):
    """
    Enhance audio using a pipeline of ClearVoice models.
    
    Args:
        input_path (str): Path to the input audio file
        output_path (str): Path where the final enhanced audio will be saved
        model_pipeline (list of dict): List of dictionaries containing 'task' and 'model_name' for each step
        temp_dir (str): Directory to store intermediate files
    
    Returns:
        str: Path to the enhanced audio file
    """
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Track the current file being processed
    current_file = input_path
    
    # Iterate through the model pipeline
    for i, model_info in enumerate(model_pipeline):
        task = model_info['task']
        model_name = model_info['model_name']
        
        logger.info(f"Step {i+1}: {task} using {model_name}")
        
        # Generate a unique intermediate file name
        intermediate_path = os.path.join(temp_dir, f'intermediate_{i}_{model_name}.wav')
        
        # Process the audio with the current model
        process_audio(task, model_name, current_file, intermediate_path)
        
        # Update the current file to the intermediate file
        current_file = intermediate_path
    
    # Move the final result to the output path
    shutil.copy(current_file, output_path)
    
    logger.info(f"Final enhanced audio saved to {output_path}")
    return output_path

def process_audio(task, model_name, input_path, output_path):
    """
    Process audio using a specific ClearVoice model.
    
    Args:
        task (str): The task to perform ('speech_enhancement', 'speech_super_resolution', etc.)
        model_name (str): The name of the model to use
        input_path (str): Path to the input audio file
        output_path (str): Path where the processed audio will be saved
    
    Returns:
        str: Path to the processed audio file
    """
    try:
        cv = ClearVoice(task=task, model_names=[model_name])
        processed_wav = cv(input_path=input_path, online_write=False)
        cv.write(processed_wav, output_path=output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error processing audio with {model_name}: {str(e)}")
        raise

def get_default_pipeline():
    """
    Returns the default model pipeline if none is specified.
    """
    return [
        {'task': 'speech_enhancement', 'model_name': 'MossFormer2_SE_48K'},
        {'task': 'speech_super_resolution', 'model_name': 'MossFormer2_SR_48K'},
        {'task': 'speech_enhancement', 'model_name': 'MossFormer2_SE_48K'}
    ]

def read_audio_from_url(url, local_path):
    """
    Download audio from a URL to a local path.
    """
    import requests
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return local_path
    except Exception as e:
        logger.error(f"Error downloading audio from URL: {str(e)}")
        raise

def read_base64_audio(base64_str, local_path):
    """
    Decode a base64 audio string and save to a local path.
    """
    try:
        if base64_str is None:
            raise ValueError("Base64 string is None")
            
        with open(local_path, "wb") as f:
            f.write(base64.b64decode(base64_str))
        return local_path
    except Exception as e:
        logger.error(f"Error decoding base64 audio: {str(e)}")
        raise

def ensure_wav_format(input_path, temp_dir='temp'):
    """
    Convert audio to WAV format if needed.
    
    Args:
        input_path (str): Path to the input audio file
        temp_dir (str): Directory to store temporary files
    
    Returns:
        str: Path to the WAV file
    """
    file_extension = os.path.splitext(input_path)[1].lower()
    
    if file_extension == '.wav':
        return input_path
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create output path for converted WAV
    wav_filename = os.path.basename(input_path).rsplit('.', 1)[0] + '.wav'
    wav_path = os.path.join(temp_dir, wav_filename)
    
    # Convert to WAV using ffmpeg
    try:
        subprocess.check_call([
            'ffmpeg', '-y', '-i', input_path, 
            '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1',
            wav_path
        ])
        return wav_path
    except subprocess.SubprocessError as e:
        logger.error(f"Error converting to WAV: {e}")
        return input_path  # Return original path as fallback

def handler(event):
    """
    RunPod serverless handler function.
    
    Expected input format:
    {
        "input": {
            "audio": "base64_encoded_audio_or_url",          # Base64 encoded audio
            "is_url": false,                                 # Set to true if providing a URL
            "audio_file": {"local_path": "/path/to/file"},   # Set by RunPod when file is uploaded
            "model_pipeline": [                              # Optional, will use default if not provided
                {"task": "speech_enhancement", "model_name": "MossFormer2_SE_48K"},
                {"task": "speech_super_resolution", "model_name": "MossFormer2_SR_48K"}
            ],
            "return_type": "base64"                          # "base64" or "file" (default: "base64")
        }
    }
    
    Returns:
    {
        "output": {
            "enhanced_audio": "base64_encoded_enhanced_audio", # If return_type is "base64"
            "file_path": "/path/to/output.wav",                # If return_type is "file"
            "pipeline_used": [list of models used]
        }
    }
    """
    try:
        # Create temporary directories for input and output
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, "input_audio.wav")
        output_path = os.path.join(temp_dir, "enhanced_audio.wav")
        
        # Get input parameters
        input_data = event.get("input", {})
        model_pipeline = input_data.get("model_pipeline", get_default_pipeline())
        return_type = input_data.get("return_type", "base64")  # Default to base64
        
        # Log input configuration
        logger.info(f"Processing request with pipeline: {json.dumps(model_pipeline)}")
        
        # Determine the input source and process accordingly
        if "audio_file" in input_data and isinstance(input_data["audio_file"], dict) and "local_path" in input_data["audio_file"]:
            # Direct file upload from RunPod
            file_path = input_data["audio_file"]["local_path"]
            logger.info(f"Using uploaded file from path: {file_path}")
            # Convert to WAV if needed
            input_path = ensure_wav_format(file_path, temp_dir)
        elif "input_url" in input_data and input_data["input_url"]:
            # URL input
            url = input_data["input_url"]
            logger.info(f"Downloading audio from URL: {url}")
            download_path = os.path.join(temp_dir, "downloaded_file")
            read_audio_from_url(url, download_path)
            # Convert to WAV if needed
            input_path = ensure_wav_format(download_path, temp_dir)
        elif "audio" in input_data and input_data["audio"]:
            # Base64 input
            logger.info(f"Decoding base64 audio")
            read_base64_audio(input_data["audio"], input_path)
        else:
            return {"error": "No audio input provided. Please provide 'audio_file', 'input_url', or 'audio'"}
        
        # Enhance the audio
        enhance_audio(input_path, output_path, model_pipeline, temp_dir=os.path.join(temp_dir, "intermediates"))
        
        # Prepare the response based on the return type
        response = {
            "output": {
                "pipeline_used": model_pipeline
            }
        }
        
        if return_type == "base64":
            # Encode the output audio to base64
            with open(output_path, "rb") as f:
                encoded_audio = base64.b64encode(f.read()).decode("utf-8")
            response["output"]["enhanced_audio"] = encoded_audio
        else:
            # Return the file path
            # Note: In RunPod, this path can be used in subsequent API calls or webhook handlers
            final_output_path = os.path.join("/tmp", f"enhanced_audio_{os.path.basename(output_path)}")
            shutil.copy(output_path, final_output_path)
            response["output"]["file_path"] = final_output_path
        
        # Don't clean up temp files right away if returning file path
        if return_type != "file":
            shutil.rmtree(temp_dir)
        
        return response
    
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        import traceback
        trace = traceback.format_exc()
        logger.error(f"Traceback: {trace}")
        return {"error": str(e), "traceback": trace}

# Start the RunPod serverless handler
if __name__ == "__main__":
    logger.info("Starting ClearerVoice-Studio RunPod serverless handler")
    runpod.serverless.start({"handler": handler})