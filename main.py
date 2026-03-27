import os
# Fix for OMP error on Windows (Must be before other imports)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import logging
import json
from transcription import AudioTranscriber
from visual_analysis import VisualAnalyzer
from moment_selection import MomentSelector
from video_editor import VideoEditor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="🎥 Creator Shorts: Automated Podcast Highlights CLI")
    
    # Required Arguments
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the source video file")
    
    # Optional Arguments
    parser.add_argument("--output", "-o", type=str, default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--mode", "-m", type=str, choices=["Fast", "High Quality"], default="Fast", help="Execution mode")
    parser.add_argument("--whisper", "-w", type=str, default="large-v3", help="Whisper model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--llm", "-l", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Local LLM model ID for moment selection")
    parser.add_argument("--llava", type=str, default="llava-hf/llava-1.5-7b-hf", help="LLaVA model ID for visual analysis (HQ mode only)")
    parser.add_argument("--no-merge", action="store_true", help="Do not merge clips into a final compilation")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cuda or cpu)")

    args = parser.parse_args()

    # Absolute Paths
    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    # 1. Initialize Models
    logger.info(f"🚀 Initializing Creator Shorts in {args.mode} mode...")
    transcriber = AudioTranscriber(model_size=args.whisper, device=args.device)
    selector = MomentSelector(model_id=args.llm, device=args.device)
    editor = VideoEditor()

    analyzer = None
    if args.mode == "High Quality":
        logger.info("Loading LLaVA for visual analysis...")
        analyzer = VisualAnalyzer(model_id=args.llava, device=args.device)

    # 2. Transcribe
    logger.info("Step 1: Transcribing audio...")
    transcription = transcriber.transcribe(input_path)
    
    # 3. Visual Analysis (Optional)
    visual_context = []
    if args.mode == "High Quality" and analyzer:
        logger.info("Step 2: Performing visual analysis (Sampling every 10s)...")
        duration_s = transcription['segments'][-1]['end']
        for ts in range(0, int(duration_s), 10):
            frame = analyzer.extract_frame(input_path, ts * 1000)
            analysis = analyzer.analyze_frame(frame)
            visual_context.append({"timestamp": ts, "analysis": analysis})

    # 4. Select Moments
    logger.info("Step 3: Selecting viral moments with LLM...")
    moments = selector.select_moments(transcription, visual_context if args.mode == "High Quality" else None)
    
    if not moments:
        logger.warning("No moments identified by the LLM.")
        return

    logger.info(f"Found {len(moments)} moments. Proceeding to clipping...")

    # 5. Generate Clips and Metadata
    clip_paths = []
    total_clips = len(moments)
    for i, m in enumerate(moments):
        clip_name = f"clip_{i}"
        mp4_path = os.path.join(output_dir, f"{clip_name}.mp4")
        json_path = os.path.join(output_dir, f"{clip_name}.json")
        
        logger.info(f"Processing clip {i+1}/{total_clips}: {clip_name}")
        
        if editor.cut_video(input_path, mp4_path, m['start'], m['end']):
            clip_paths.append(mp4_path)
            
            # Detailed JSON
            clip_metadata = {
                "clip_id": i,
                "timing": {"start": m['start'], "end": m['end'], "duration": m['end']-m['start']},
                "analysis": {"reason": m['reason'], "climax_description": f"Destaque: {m['reason']}"},
                "batch_info": {"total_clips_generated": total_clips, "source_video": os.path.basename(input_path)}
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(clip_metadata, f, indent=4, ensure_ascii=False)

    # 6. Global Metadata
    metadata_path = os.path.join(output_dir, "batch_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(moments, f, indent=4, ensure_ascii=False)

    # 7. Final Merge
    if not args.no_merge and clip_paths:
        logger.info("Merging clips into final compilation...")
        final_path = os.path.join(output_dir, "final_compilation.mp4")
        editor.merge_videos(clip_paths, final_path)
    
    logger.info(f"✨ Process completed. Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
