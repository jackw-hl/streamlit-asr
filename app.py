import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
from tqdm import tqdm

st.title("Cantonese ASR")

# File uploader with drag-and-drop functionality
uploaded_file = st.file_uploader("Upload your video file", type=["mp4"], key="file_uploader")

if uploaded_file is not None:
    lang = "zh"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # model_id = "openai/whisper-large-v3-turbo"
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
    video_file = "test.mp4"
    audio_file = "test.wav"
    audio = AudioSegment.from_file(video_file, format="mp4")
    audio.export(audio_file, format="wav")
    # Process audio in chunks
    chunk_length_ms = 30000  # 30 seconds
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    transcription = ""
    progress_bar = st.progress(0)
    total_chunks = len(chunks)

    for i, chunk in enumerate(tqdm(chunks, desc="Processing audio")):
        chunk.export("temp_chunk.wav", format="wav")
        sample = "temp_chunk.wav"
        result = pipe(sample)
        transcription += result["text"] + " "
        progress_bar.progress((i + 1) / total_chunks)

    output_file = "transcription.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcription)
    st.success(f"Transcription saved to {output_file}")

    st.title("Transcription Result")
    with st.expander("Response"):
        st.info(transcription.strip())

    # Create a download button for the output file
    with open(output_file, "rb") as f:
        st.download_button(
            label="Download Transcription",
            data=f,
            file_name=output_file,
            mime="text/plain"
        )
    uploaded_file = None
