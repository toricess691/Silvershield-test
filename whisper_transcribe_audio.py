"""
    This receives an audio file to which we can transcribe and for our naive bayes scam classification.
"""
import whisper
import torch
from pathlib import Path
import time
import argparse

def transcribe_audio(audio_path, model_size="base"):
    """
    Transcribe an audio file using OpenAI's Whisper model.
    
    Parameters:
    -----------
    audio_path : str or Path
        Path to the audio file to transcribe
    model_size : str, optional (default="base")
        Size of the Whisper model to use. Options include:
        "tiny", "base", "small", "medium", "large"
    
    Returns:
    --------
    dict
        Dictionary containing the transcription and other metadata
    """
    # Convert string path to Path object
    audio_path = Path(audio_path).resolve()
    
    # Check if file exists
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
    
    print(f"Loading Whisper model: {model_size}")
    # Load the Whisper model
    model = whisper.load_model(model_size)
    
    # Check if CUDA (GPU) is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Start timing
    start_time = time.time()
    
    print(f"Starting transcription of: {audio_path.name}")
    # Perform the transcription
    result = model.transcribe(
        str(audio_path),
        fp16=torch.cuda.is_available()  # Use half-precision if GPU is available
    )
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Add processing time to results
    result['processing_time'] = processing_time
    
    return result

def save_transcript(transcript_data, output_path):
    """
    Save the transcription to a text file.
    
    Parameters:
    -----------
    transcript_data : dict
        Dictionary containing the transcription data
    output_path : str or Path
        Path where the transcript should be saved
    """
    output_path = Path(output_path)
    
    # Create the output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the transcript to a file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(transcript_data['text'])
        
    print(f"Transcript saved to: {output_path}")
    print(f"Processing time: {transcript_data['processing_time']:.2f} seconds")

def main():
    """
    Process command line arguments and run the transcription.
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Transcribe audio files using Whisper')
    parser.add_argument('audio_file', type=str, help='Path to the audio file to transcribe')
    parser.add_argument('--output', '-o', type=str, default=None,
                      help='Output file path (default: input_filename_transcript.txt)')
    parser.add_argument('--model', '-m', type=str, default='base',
                      choices=['tiny', 'base', 'small', 'medium', 'large'],
                      help='Whisper model size to use (default: base)')
    
    # Parse the arguments
    args = parser.parse_args()
    
    try:
        # Get the input file path
        audio_path = Path(args.audio_file)
        
        # If no output path specified, create one based on input filename
        if args.output is None:
            output_path = audio_path.parent / f"{audio_path.stem}_transcript.txt"
        else:
            output_path = Path(args.output)
        
        # Transcribe the audio
        transcript = transcribe_audio(audio_path, model_size=args.model)
        
        # Save the transcript
        save_transcript(transcript, output_path)
        
        # Print some information about the transcription
        print("\nTranscription Summary:")
        print("----------------------")
        print(f"Model used: {transcript.get('model_name', 'unknown')}")
        print(f"Language detected: {transcript.get('language', 'unknown')}")
        print(f"Transcription length: {len(transcript['text'])} characters")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print(f"Please check if the file exists and the path is correct.")
        print(f"Current working directory: {Path.cwd()}")
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        print("If you haven't installed the required packages, please run:")
        print("pip install openai-whisper torch")

if __name__ == "__main__":
    main()