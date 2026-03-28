import os

# Fix for OMP error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, model_size="large-v3", device="auto", compute_type="float16"):
        """
        Initializes the Faster-Whisper model.
        device: 'cuda', 'cpu', or 'auto'
        compute_type: 'float16' for GPU, 'int8' or 'default' for CPU
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        
        # Adjust compute_type if device is CPU
        if device == "cpu":
            self.compute_type = "int8"
            
        logger.info(f"Loading Faster-Whisper model '{model_size}' on {device}...")
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        logger.info("Model loaded.")

    def transcribe(self, audio_path, language=None):
        """
        Transcribes the audio and returns segments with timestamps.
        """
        logger.info(f"Transcribing: {audio_path}")
        # Use VAD (Voice Activity Detection) to skip silence and speed up transcription
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        results = []
        # Duration info
        logger.info(f"Processing audio with duration {info.duration_after_vad:.2f}s (VAD filtered)")
        for segment in segments:
            res = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": []
            }
            if segment.words:
                for word in segment.words:
                    res["words"].append({
                        "word": word.word.strip(),
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    })
            results.append(res)
            
        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": results
        }
