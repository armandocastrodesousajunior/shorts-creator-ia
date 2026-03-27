# Creator Shorts 🎥

Automatização de short videos a partir de podcasts utilizando IA (Whisper, LLaVA, LLM Local).

## Funcionalidades
- **Transcrição**: Utiliza OpenAI Whisper para converter áudio em texto com timestamps.
- **Modos de Execução**: 
    - **Fast**: Apenas transcrição para identificar cortes.
    - **High Quality**: Transcrição + Análise de frames (LLaVA) para momentos visuais impactantes.
- **Moment Selection**: Um LLM local analisa o contexto e sugere os melhores "hooks".
- **Edição Automática**: Cortes precisos via FFmpeg, com opção de merge.

## Instalação (Conda)
```bash
conda env create -f environment.yml
conda activate creator-shorts
```

## Arquitetura do Projeto
O sistema está dividido em módulos especializados:
- **`transcription.py`**: Transcrição veloz com `faster-whisper`.
- **`visual_analysis.py`**: Análise de frames com `LLaVA-1.5-7B`.
- **`moment_selection.py`**: Identificação de momentos virais com `Mistral-7B` ou `Llama-3`.
- **`video_editor.py`**: Edição e merge de vídeos via `FFmpeg`.

## Como Usar

### 1. Interface de Linha de Comando (CLI)
Você pode processar qualquer vídeo diretamente pelo terminal:
```bash
conda activate creator-shorts
python main.py --input "caminho/do/video.mp4" --output "pasta_de_saida" --mode Fast
```

**Principais Argumentos:**
- `--input` / `-i`: Caminho do vídeo original (obrigatório).
- `--output` / `-o`: Pasta onde os cortes e JSONs serão salvos.
- `--mode` / `-m`: `Fast` (vapt-vupt) ou `High Quality` (mais lento, com análise visual).
- `--whisper` / `-w`: Tamanho do modelo Whisper (`tiny`, `base`, `small`, `medium`, `large-v3`).
- `--llm` / `-l`: ID do modelo LLM local (ex: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).

### 2. Jupyter Notebook / Google Colab
Abra o arquivo `shorts-creator-ia.ipynb` para uma experiência interativa e visual.

## Requisitos Técnicos
- **GPU**: Recomendado 8GB+ de VRAM para rodar o modo "Alta Qualidade" localmente.
- **FFmpeg**: Deve estar instalado no sistema.
