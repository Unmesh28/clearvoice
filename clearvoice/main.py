from clearvoice import ClearVoice
import os
import shutil

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

if __name__ == "__main__":
    input_path = 'D:\My_Work\Podcast\Speech_Enhance\download2.wav'  # Replace with your input audio file
    output_path = 'D:\My_Work\Podcast\Speech_Enhance\output_download2.wav'  # Replace with your desired output path
    
    # Define the model pipeline (sequence of models and tasks)
    model_pipeline = [
        #   {'task': 'speech_enhancement', 'model_name': 'FRCRN_SE_16K'},
          {'task': 'speech_enhancement', 'model_name': 'MossFormer2_SE_48K'},
        #   {'task': 'speech_enhancement', 'model_name': 'MossFormerGAN_SE_16K'},
          {'task': 'speech_super_resolution', 'model_name': 'MossFormer2_SR_48K'},
          {'task': 'speech_enhancement', 'model_name': 'MossFormer2_SE_48K'}
    ]
    
    # Enhance the audio using the specified model pipeline
    enhance_audio(input_path, output_path, model_pipeline)