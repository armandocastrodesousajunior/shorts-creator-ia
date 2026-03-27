import os
# Fix for OMP error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import logging
import json

from transcription import AudioTranscriber
from moment_selection import MomentSelector
from video_editor import VideoEditor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_real_test(video_path):
    # Use absolute path for reliability
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "real_podcast_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize
    logger.info("Initializing models for REAL Podcast Test (Fast Mode)...")
    transcriber = AudioTranscriber(model_size="base") # Fast transcription
    
    # Using TinyLlama to fit in 6GB VRAM
    selector = MomentSelector(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # 2. Transcribe
    logger.info(f"Step 1: Transcribing {video_path}...")
    transcription = transcriber.transcribe(video_path)
    
    # 3. Select Moments (REAL LLM)
    logger.info("Step 2: Selecting moments with real LLM analysis...")
    moments = selector.select_moments(transcription)
    
    if not moments:
        logger.warning("LLM failed to identify moments. Using a fallback segment for pipeline verification.")
        moments = [{"start": 30.0, "end": 90.0, "reason": "Segmento de backup para teste"}]
    
    # 4. Clipping and Metadata Export
    logger.info(f"Step 3: Generating {len(moments)} clips and detailed JSON...")
    editor = VideoEditor()
    clip_paths = []
    total_clips = len(moments)
    
    for i, m in enumerate(moments):
        clip_name = f"podcast_real_{i}"
        mp4_path = os.path.join(output_dir, f"{clip_name}.mp4")
        json_path = os.path.join(output_dir, f"{clip_name}.json")
        
        # Cut Video (-c copy for speed and reliability)
        if editor.cut_video(video_path, mp4_path, m['start'], m['end']):
            clip_paths.append(mp4_path)
            
            # Generate Detailed JSON
            clip_metadata = {
                "clip_id": i,
                "timing": {"start": m['start'], "end": m['end'], "duration": m['end'] - m['start']},
                "analysis": {"reason": m['reason'], "climax_description": f"Destaque: {m['reason']}"},
                "batch_info": {"total_clips_generated": total_clips, "source_video": os.path.basename(video_path)}
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(clip_metadata, f, indent=4, ensure_ascii=False)
            logger.info(f"Generated clip and metadata: {clip_name}")

    # 5. Global Metadata
    metadata_path = os.path.join(output_dir, "final_batch_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(moments, f, indent=4, ensure_ascii=False)
            
    # 6. Merge
    if clip_paths:
        final_path = os.path.join(output_dir, "podcast_real_compilation.mp4")
        editor.merge_videos(clip_paths, final_path)
    
    logger.info(f"REAL test complete. Results in: {output_dir}")

if __name__ == "__main__":
    v_path = r"C:\Users\arman\Downloads\PODCAST.mp4"
    run_real_test(v_path)
