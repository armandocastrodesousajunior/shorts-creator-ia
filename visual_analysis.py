import torch
from transformers import pipeline
from PIL import Image
import cv2
import os
# Fix for OMP error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging

logger = logging.getLogger(__name__)

class VisualAnalyzer:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device="cuda"):
        """
        Initializes LLaVA model for visual analysis.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading LLaVA model '{model_id}' on {self.device}...")
        
        # Load pipeline for image-to-text
        self.pipe = pipeline(
            "image-to-text", 
            model=model_id, 
            device=0 if self.device == "cuda" else -1,
            model_kwargs={"torch_dtype": torch.float16 if self.device == "cuda" else torch.float32}
        )
        logger.info("LLaVA model loaded.")

    def extract_frame(self, video_path, timestamp_ms):
        """
        Extracts a frame from the video at a specific timestamp.
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        success, image = cap.read()
        cap.release()
        
        if success:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image_rgb)
        return None

    def analyze_frame(self, frame_image, prompt="Describe the facial expressions and emotions in this frame. Is there any intense reaction or interesting gesture?"):
        """
        Analyzes a frame using LLaVA.
        """
        if frame_image is None:
            return "No frame available"
            
        # Standard LLaVA prompt format
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        outputs = self.pipe(frame_image, prompt=full_prompt, generate_kwargs={"max_new_tokens": 100})
        
        return outputs[0]["generated_text"].split("ASSISTANT:")[-1].strip()
