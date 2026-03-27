import subprocess
import os
import logging

logger = logging.getLogger(__name__)

class VideoEditor:
    def __init__(self, ffmpeg_path="ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

    def cut_video(self, input_path, output_path, start_time, end_time):
        """
        Cuts a segment from the video using FFmpeg.
        """
        duration = end_time - start_time
        command = [
            self.ffmpeg_path,
            "-y",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "copy",
            "-c:a", "copy",
            output_path
        ]
        
        logger.info(f"Cutting video: {start_time} to {end_time} -> {output_path}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
        return True

    def merge_videos(self, video_paths, output_path):
        """
        Merges multiple video clips into one.
        """
        if not video_paths:
            return False
            
        # Create a temp file list for ffmpeg concat
        list_file = "temp_concat_list.txt"
        with open(list_file, "w") as f:
            for path in video_paths:
                f.write(f"file '{path}'\n")
        
        command = [
            self.ffmpeg_path,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path
        ]
        
        logger.info(f"Merging videos into {output_path}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Cleanup
        if os.path.exists(list_file):
            os.remove(list_file)
            
        if result.returncode != 0:
            logger.error(f"FFmpeg merge error: {result.stderr}")
            return False
        return True
