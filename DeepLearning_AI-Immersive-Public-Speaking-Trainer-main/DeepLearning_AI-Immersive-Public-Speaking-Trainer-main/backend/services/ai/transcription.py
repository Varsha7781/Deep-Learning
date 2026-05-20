import os
import whisper
import torch

class TranscriptionService:
    """
    Handles Speech-to-Text using OpenAI's Whisper model.
    Optimized for capturing speech nuances like filler words and pauses.
    """
    def __init__(self, model_name: str = "base"):
        """Initialize the Whisper model for STT."""
        try:
            print(f"Loading Whisper model: {model_name}...")
            self.model = whisper.load_model(model_name)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.model = None

    def transcribe(self, audio_path: str):
        """
        Transcribe audio file to text.
        Includes initial_prompt to capture filler words (disfluencies) more accurately.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            result = self.model.transcribe(
                audio_path, 
                verbose=False,
                word_timestamps=True # Enable for more granular timing if needed
            )
            
            if not result["text"].strip():
                print("⚠ No speech detected in audio")
            return {
                "text": result["text"].strip(),
                "segments": result["segments"], # Contains 'start', 'end', 'text' 
                "language": result["language"]
            }
        except Exception as e:
            print(f"Transcription Error: {str(e)}")
            raise e
