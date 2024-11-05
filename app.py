import torch
import streamlit as st
import io
import os

from pydub import AudioSegment
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

st.title("Cantonese ASR Transcription")

# Initialize session state for tracking processing status
if 'processed' not in st.session_state:
    st.session_state.processed = False

# File uploader with drag-and-drop functionality
uploaded_file = st.file_uploader("Upload your video file", type=["mp4"], key="file_uploader")

# Reset the processed flag if a new file is uploaded
if uploaded_file is not None:
    st.session_state.processed = False

if uploaded_file is not None and not st.session_state.processed:
    lang = "zh"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "alvanlii/whisper-small-cantonese"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device
    )
    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
    # Read the uploaded file as bytes
    audio_bytes = uploaded_file.read()
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp4")

    # Process audio in chunks
    chunk_length_ms = 1000  # 1 seconds
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    transcription = ""
    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_chunks = len(chunks)
    progress_text.text(f"Progress: {0}%")
    for i, chunk in enumerate(tqdm(chunks, desc="Processing audio")):
        result = pipe(chunk.export(format="wav").read())
        transcription += result['text'] + " "
        progress = (i + 1) / total_chunks
        progress_bar.progress(progress)
        progress_text.text(f"Progress: {int(progress * 100)}%")

    st.title("Result")
    with st.expander("Transcription"):
        st.info(transcription)

    # Get the filename without the extension
    filename_without_extension = os.path.splitext(uploaded_file.name)[0]

    # Use the filename without the extension for the output file name
    output_file_name = f"{filename_without_extension}_transcription.txt"

    # Create a download button for the output file
    st.download_button(
        label="Download Transcription",
        file_name=output_file_name,
        mime="text/plain",
        data=transcription.encode('utf-8')
    )

    # Mark the file as processed
    st.session_state.processed = True