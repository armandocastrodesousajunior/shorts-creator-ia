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

        # Divide into chunks of ~2-3 minutes or 40 segments to fit TinyLlama's 2048 context window
        chunk_size = 40 
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
            Task: Identify interesting viral moments in this transcript segment.
            Constraint: Return ONLY a JSON list of objects. No intro, no outro.
            Format: [{{"start": 0.0, "end": 30.0, "reason": "summary"}}]
            
            Transcript Fragment:
            {content}
            
            Answer (JSON ONLY):
            """
            
            try:
                logger.info(f"Querying LLM for chunk {i//chunk_size + 1}...")
                outputs = self.pipe(prompt)
                
                # Get response and handle prompt repetition
                raw_response = outputs[0]["generated_text"]
                if "Answer (JSON ONLY):" in raw_response:
                    response_text = raw_response.split("Answer (JSON ONLY):")[-1].strip()
                else:
                    # Fallback if the whole prompt is returned
                    response_text = raw_response.strip()

                # Robust JSON extraction: Find the first '[' and the last ']'
                start_idx = response_text.find("[")
                end_idx = response_text.rfind("]")
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx+1]
                    # Clean up common LLM errors like trailing commas
                    json_str = json_str.replace(",]", "]").replace(", }", "}")
                    
                    moments = json.loads(json_str)
                    if isinstance(moments, list):
                        all_moments.extend(moments)
                else:
                    logger.warning(f"No JSON list found in chunk {i//chunk_size + 1} response.")
                    
            except Exception as e:
                logger.warning(f"Failed to process chunk {i//chunk_size + 1}: {e}")
                continue

        # Limit and deduplicate nearby moments
        return all_moments[:12]
