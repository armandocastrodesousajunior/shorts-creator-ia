import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
# Fix for OMP error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import logging

logger = logging.getLogger(__name__)

class MomentSelector:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2", device="cuda"):
        """
        Initializes a local LLM for identifying viral moments.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading LLM '{model_id}' on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
        logger.info("LLM loaded.")

    def select_moments(self, transcription_results, visual_context=None):
        """
        Analyzes transcription (and visual context) to find interesting segments.
        Expects transcription_results to have 'segments' list.
        """
        # Prepare content for LLM
        content = ""
        for seg in transcription_results['segments']:
            content += f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n"
            
        if visual_context:
            content += "\nVisual Context cues:\n"
            for ctx in visual_context:
                content += f"At {ctx['timestamp']:.2f}s: {ctx['analysis']}\n"

        prompt = f"""
        Analyze the following podcast transcript and identify the most viral, interesting, or insightful moments for short-form video clips (30-60 seconds each).
        Return your answer ONLY as a JSON list of objects with 'start', 'end', and 'reason' keys.
        
        Format example:
        [
          {{"start": 12.5, "end": 45.0, "reason": "Funny story about AI"}},
          {{"start": 120.0, "end": 150.0, "reason": "Deep insight about the future"}}
        ]
        
        Transcript:
        {content}
        
        Answer (JSON only):
        """
        
        logger.info("Querying LLM for moment selection...")
        outputs = self.pipe(prompt)
        response_text = outputs[0]["generated_text"].split("Answer (JSON only):")[-1].strip()
        
        # Try to parse JSON
        try:
            # Clean potential Markdown backticks
            if "```json" in response_text:
                response_text = response_text.split("```json")[-1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[-1].split("```")[0].strip()
                
            moments = json.loads(response_text)
            return moments
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {response_text}")
            return []
