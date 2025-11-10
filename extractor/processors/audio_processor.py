"""
Audio processing module for Whisper transcription.
"""

import os
import time
import tempfile
import subprocess
from typing import Dict, Optional
from extractor.models.whisper_loader import WhisperLoader
from extractor.config import WHISPER_MODEL


class AudioProcessor:
    """Processes audio segments with Whisper."""
    
    def __init__(self, whisper_loader: WhisperLoader, video_path: str):
        """
        Initialize audio processor.
        
        Args:
            whisper_loader: Whisper model loader instance
            video_path: Path to video file
        """
        self.whisper_loader = whisper_loader
        self.video_path = video_path
    
    def transcribe_segment(self, start_sec: float, end_sec: float) -> Dict:
        """
        Transcribe audio for a specific time segment.
        
        Args:
            start_sec: Start time in seconds
            end_sec: End time in seconds
            
        Returns:
            Dictionary with transcription and metadata
        """
        if not self.whisper_loader.is_loaded():
            return {
                "transcription": "",
                "transcription_metadata": {
                    "model": WHISPER_MODEL,
                    "language": "en",
                    "processing_time": 0,
                    "error": "Whisper model not initialized"
                }
            }
        
        start_time = time.time()
        temp_audio_path = None
        
        try:
            # Create temporary file for audio segment
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_audio_path = tmp_file.name
            
            # Extract audio segment with ffmpeg
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', self.video_path,
                '-ss', str(start_sec),
                '-t', str(end_sec - start_sec),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # Whisper works well with 16kHz
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                temp_audio_path
            ]
            
            # Run ffmpeg
            subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True
            )
            
            # Validate audio file
            if not os.path.exists(temp_audio_path):
                raise Exception("Audio file was not created by FFmpeg")
            
            file_size = os.path.getsize(temp_audio_path)
            if file_size < 500:
                raise Exception(f"Audio file too small ({file_size} bytes), likely empty or invalid")
            
            # Transcribe with Whisper
            transcription_text = ""
            whisper_result = None
            try:
                whisper_result = self.whisper_loader.transcribe(
                    temp_audio_path,
                    language="en",
                    task="transcribe"
                )
                transcription_text = whisper_result["text"].strip()
            except Exception as whisper_error:
                if "reshape" in str(whisper_error).lower() or "0 elements" in str(whisper_error):
                    transcription_text = ""  # Empty audio
                else:
                    raise
            
            processing_time = time.time() - start_time
            
            return {
                "transcription": transcription_text,
                "transcription_metadata": {
                    "model": WHISPER_MODEL,
                    "language": whisper_result.get("language", "en") if whisper_result else "en",
                    "processing_time": round(processing_time, 2)
                }
            }
        
        except subprocess.CalledProcessError as e:
            processing_time = time.time() - start_time
            return {
                "transcription": "",
                "transcription_metadata": {
                    "model": WHISPER_MODEL,
                    "language": "en",
                    "processing_time": round(processing_time, 2),
                    "error": f"FFmpeg error: {str(e)}"
                }
            }
        except FileNotFoundError:
            processing_time = time.time() - start_time
            return {
                "transcription": "",
                "transcription_metadata": {
                    "model": WHISPER_MODEL,
                    "language": "en",
                    "processing_time": round(processing_time, 2),
                    "error": "FFmpeg not found. Please install FFmpeg."
                }
            }
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "transcription": "",
                "transcription_metadata": {
                    "model": WHISPER_MODEL,
                    "language": "en",
                    "processing_time": round(processing_time, 2),
                    "error": str(e)
                }
            }
        finally:
            # Clean up temporary file
            try:
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            except:
                pass

