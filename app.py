import os
import yt_dlp

# Fix for OMP error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
import logging
import json
from transcription import AudioTranscriber
from visual_analysis import VisualAnalyzer
from moment_selection import MomentSelector
from video_editor import VideoEditor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_youtube_video(url, output_dir="gradio_outputs", progress=None):
    os.makedirs(output_dir, exist_ok=True)
    
    def progress_hook(d):
        if d['status'] == 'downloading' and progress:
            p = d.get('downloaded_bytes', 0) / d.get('total_bytes', 1)
            progress(p * 0.1, desc=f"📥 Baixando do YouTube: {d['_percent_str']}")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,
        'progress_hooks': [progress_hook],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

def process_video(video_path, yt_url, mode, whisper_size, llm_id, progress=gr.Progress()):
    if not video_path and not yt_url:
        return None, None, "Por favor, envie um vídeo ou um link do YouTube."

    output_dir = "gradio_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Download se for URL com progresso
        final_video_path = video_path
        if yt_url:
            final_video_path = download_youtube_video(yt_url, output_dir, progress)

        progress(0.1, desc="🚀 Inicializando modelos...")
        transcriber = AudioTranscriber(model_size=whisper_size)
        selector = MomentSelector(model_id=llm_id)
        editor = VideoEditor()
        
        analyzer = None
        if mode == "High Quality":
            analyzer = VisualAnalyzer()

        progress(0.2, desc="🎙️ Transcrevendo áudio...")
        transcription = transcriber.transcribe(final_video_path)
        
        visual_context = []
        if mode == "High Quality":
            progress(0.4, desc="👁️ Analisando visualmente...")
            # Dynamically adjust sampling interval based on duration
            try:
                duration_s = transcription['segments'][-1]['end']
            except (KeyError, IndexError):
                duration_s = 60 # Default fallback
                
            # Sample at most ~100 frames to avoid hang, or every 60s for long videos
            interval = max(60, int(duration_s / 100)) 
            
            for ts in range(0, int(duration_s), interval):
                frame = analyzer.extract_frame(final_video_path, ts * 1000)
                analysis = analyzer.analyze_frame(frame)
                visual_context.append({"timestamp": ts, "analysis": analysis})
        
        progress(0.6, desc="🧠 Selecionando melhores momentos...")
        moments = selector.select_moments(transcription, visual_context if mode == "High Quality" else None)
        
        if not moments:
            return None, None, "Nenhum momento viral identificado."

        progress(0.8, desc="✂️ Gerando clipes...")
        clip_paths = []
        metadata_list = []
        
        for i, m in enumerate(moments):
            clip_name = f"clip_{i}.mp4"
            clip_out = os.path.join(output_dir, clip_name)
            if editor.cut_video(final_video_path, clip_out, m['start'], m['end']):
                clip_paths.append(clip_out)
                metadata_list.append({
                    "moment": i,
                    "reason": m['reason'],
                    "start": f"{m['start']:.2f}s",
                    "end": f"{m['end']:.2f}s"
                })

        final_compilation = None
        if clip_paths:
            progress(0.9, desc="🔗 Unindo clipes...")
            final_compilation = os.path.join(output_dir, "final_compilation.mp4")
            editor.merge_videos(clip_paths, final_compilation)

        progress(1.0, desc="✅ Concluído!")
        return final_compilation, clip_paths, json.dumps(metadata_list, indent=4, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Erro no processamento: {str(e)}")
        return None, None, f"Erro: {str(e)}"

# Custom Theme
theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

with gr.Blocks(theme=theme, title="Creator Shorts IA") as demo:
    gr.Markdown("""
    # 🎥 Creator Shorts IA
    ### Transforme seus Podcasts em Cortes Virais automaticamente.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="Upload do Podcast (Opcional se usar Link)")
            yt_url = gr.Textbox(label="OU Link do YouTube", placeholder="https://www.youtube.com/watch?v=...")
            
            mode_radio = gr.Radio(["Fast", "High Quality"], label="Modo", value="Fast")
            
            with gr.Accordion("Configurações Avançadas", open=False):
                whisper_size = gr.Dropdown(
                    ["tiny", "base", "small", "medium", "large-v3"], 
                    label="Modelo Whisper", 
                    value="small"
                )
                llm_id = gr.Dropdown(
                    ["mistralai/Mistral-7B-Instruct-v0.2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"], 
                    label="Modelo LLM (Cérebro)", 
                    value="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                )
            
            btn_run = gr.Button("🚀 Gerar Cortes", variant="primary")
            
        with gr.Column(scale=1):
            output_final = gr.Video(label="Compilação Final")
            output_gallery = gr.Gallery(label="Clipes Individuais", columns=2)
            output_json = gr.Code(label="Metadados dos Cortes", language="json")

    btn_run.click(
        process_video,
        inputs=[input_video, yt_url, mode_radio, whisper_size, llm_id],
        outputs=[output_final, output_gallery, output_json]
    )
    
    gr.Markdown("--- \n *Dica: Use o link do YouTube para evitar uploads lentos de arquivos grandes!*")

if __name__ == "__main__":
    demo.launch(share=True)
