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
        Analyzes transcription to find interesting segments.
        Extremely robust version for small models like TinyLlama.
        """
        import re
        segments = transcription_results.get('segments', [])
        if not segments:
            return []

        chunk_size = 40 
        all_moments = []
        
        logger.info(f"Processing {len(segments)} segments in chunks of {chunk_size}...")
        
        for i in range(0, len(segments), chunk_size):
            chunk = segments[i:i + chunk_size]
            content = ""
            for seg in chunk:
                content += f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}\n"

            # Extreme simplicity for TinyLlama
            prompt = f"<|system|>\nList viral moments as JSON. Format: [{{ \"start\": T1, \"end\": T2 }}]<|user|>\nTranscript:\n{content}\n<|assistant|>\nJSON:"
            
            try:
                logger.info(f"Querying chunk {i//chunk_size + 1}...")
                outputs = self.pipe(prompt)
                raw_response = outputs[0]["generated_text"]
                
                # Split by assistant marker to get only the answer
                if "<|assistant|>\nJSON:" in raw_response:
                    response_text = raw_response.split("<|assistant|>\nJSON:")[-1].strip()
                elif "JSON:" in raw_response:
                    response_text = raw_response.split("JSON:")[-1].strip()
                else:
                    response_text = raw_response.strip()

                # 1. Try standard JSON parsing
                start_idx = response_text.find("[")
                end_idx = response_text.rfind("]")
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx+1]
                    try:
                        moments = json.loads(json_str)
                        if isinstance(moments, list):
                            all_moments.extend(moments)
                            continue
                    except:
                        pass # Fallback to regex

                # 2. Regex Fallback: Find any objects like {"start": 10, "end": 20}
                matches = re.findall(r'\{\s*"start":\s*(\d+\.?\d*),\s*"end":\s*(\d+\.?\d*)[^}]*\}', response_text)
                if matches:
                    logger.info(f"Regex found {len(matches)} moments in chunk {i//chunk_size + 1}")
                    for m in matches:
                        all_moments.append({"start": float(m[0]), "end": float(m[1]), "reason": "Viral moment"})
                else:
                    # 3. Last Resort Fallback: Look for the first 30 seconds of the chunk if LLM failed
                    logger.warning(f"Chunk {i//chunk_size + 1}: LLM failed. Using heuristic fallback.")
                    all_moments.append({
                        "start": chunk[0]['start'], 
                        "end": min(chunk[0]['start'] + 45, chunk[-1]['end']), 
                        "reason": "Interesting segment"
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to process chunk {i//chunk_size + 1}: {e}")
                continue

        # --- Post-processing: Enforce min/max duration and remove overlaps ---
        MIN_DURATION = 40    # seconds (minimum clip length)
        MAX_DURATION = 180   # seconds (maximum clip length = 3 min)
        
        # 1. Basic validation and type coercion
        valid_moments = []
        for m in all_moments:
            try:
                start = float(m.get('start', -1))
                end = float(m.get('end', -1))
                if start < 0 or end <= start:
                    continue
                valid_moments.append({"start": start, "end": end, "reason": m.get("reason", "Viral moment")})
            except (TypeError, ValueError):
                continue

        # 2. Sort by start time
        valid_moments.sort(key=lambda x: x['start'])
        
        # 3. Enforce min duration by expanding the END (not forcing a fixed 60s)
        #    Only expand if shorter than minimum. Keep natural duration if >= min.
        for m in valid_moments:
            duration = m['end'] - m['start']
            if duration < MIN_DURATION:
                m['end'] = m['start'] + MIN_DURATION
            if (m['end'] - m['start']) > MAX_DURATION:
                m['end'] = m['start'] + MAX_DURATION

        # 4. Remove overlapping clips (greedy: keep first, skip if overlaps with previous)
        final_moments = []
        for m in valid_moments:
            if not final_moments:
                final_moments.append(m)
                continue
            prev = final_moments[-1]
            # Skip if this clip starts before the previous one ends
            if m['start'] < prev['end']:
                continue
            final_moments.append(m)
        
        logger.info(f"Final: {len(final_moments)} non-overlapping clips (from {len(all_moments)} candidates).")
        return final_moments[:12]
