import torch
import streamlit as st
import io
from pydub import AudioSegment
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

st.title("Cantonese ASR")

# Initialize session state for tracking processing status
if 'processed' not in st.session_state:
    st.session_state.processed = False

# File uploader with drag-and-drop functionality
uploaded_file = st.file_uploader("Upload your video file", type=["mp4"], key="file_uploader")

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

    # Read the uploaded file as bytes
    audio_bytes = uploaded_file.read()
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp4")

    # Process audio in chunks
    chunk_length_ms = 30000  # 30 seconds
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    transcription = ""
    progress_bar = st.progress(0)
    total_chunks = len(chunks)

    for i, chunk in enumerate(tqdm(chunks, desc="Processing audio")):
        result = pipe(chunk.export(format="wav").read())
        transcription += result['text'] + " "
        progress_bar.progress((i + 1) / total_chunks)

    st.title("Transcription Result")
    with st.expander("Response"):
        st.info(transcription)

    # Create a download button for the output file
    st.download_button(
        label="Download Transcription",
        file_name=f"{uploaded_file.name}_transcription.txt",
        mime="text/plain",
        data=transcription.encode('utf-8')
    )

    # Mark the file as processed
    st.session_state.processed = True