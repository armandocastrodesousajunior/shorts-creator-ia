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
        Handles long transcripts by splitting into chunks.
        """
        segments = transcription_results.get('segments', [])
        if not segments:
            return []

        # Divide into chunks of ~5 minutes or 100 segments to fit context window
        chunk_size = 80 
        all_moments = []
        
        logger.info(f"Processing {len(segments)} segments in chunks of {chunk_size}...")
        
        for i in range(0, len(segments), chunk_size):
            chunk = segments[i:i + chunk_size]
            
            # Prepare content for this chunk
            content = ""
            for seg in chunk:
                content += f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n"
            
            # Add visual context if it falls within this chunk's time range
            if visual_context:
                chunk_start = chunk[0]['start']
                chunk_end = chunk[-1]['end']
                for ctx in visual_context:
                    if chunk_start <= ctx['timestamp'] <= chunk_end:
                        content += f"At {ctx['timestamp']:.2f}s: {ctx['analysis']}\n"

            prompt = f"""
            Analyze the following podcast transcript segment and identify the most viral or interesting moments (30-60 seconds each).
            Return your answer ONLY as a JSON list of objects.
            Format example: [{{"start": 12.5, "end": 45.0, "reason": "Funny story"}}]
            
            Transcript Fragment:
            {content}
            
            Answer (JSON only):
            """
            
            try:
                logger.info(f"Querying LLM for chunk {i//chunk_size + 1}...")
                outputs = self.pipe(prompt)
                response_text = outputs[0]["generated_text"].split("Answer (JSON only):")[-1].strip()
                
                # Clean potential Markdown backticks
                if "```json" in response_text:
                    response_text = response_text.split("```json")[-1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[-1].split("```")[0].strip()
                
                # Basic JSON cleanup for aggressive LLMs
                if not response_text.startswith("["):
                    # Find first '[' and last ']'
                    start_idx = response_text.find("[")
                    end_idx = response_text.rfind("]")
                    if start_idx != -1 and end_idx != -1:
                        response_text = response_text[start_idx:end_idx+1]

                moments = json.loads(response_text)
                if isinstance(moments, list):
                    all_moments.extend(moments)
                    
            except Exception as e:
                logger.warning(f"Failed to process chunk {i//chunk_size + 1}: {e}")
                continue

        # Limit total moments to avoid excessive clipping
        return all_moments[:10]
