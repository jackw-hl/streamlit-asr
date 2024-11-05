import torch
import streamlit as st
import io
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, GenerationConfig
from pydub import AudioSegment

st.title("Cantonese ASR Transcription")

if 'processed' not in st.session_state:
    st.session_state.processed = False

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

    # Set up generation config
    generation_config = GenerationConfig.from_pretrained(model_id)
    generation_config.no_timestamps_token_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    model.generation_config = generation_config

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device
    )
    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")

    # Process audio
    audio_bytes = uploaded_file.read()
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp4")

    with st.spinner('Processing...'):
        transcription = pipe(audio.export(format="wav").read())['text']

    st.success('Done!')
    st.title("Result")
    with st.expander("Transcription"):
        st.info(transcription)

    # Create download button
    filename_without_extension = os.path.splitext(uploaded_file.name)[0]
    output_file_name = f"{filename_without_extension}_transcription.txt"

    st.download_button(
        label="Download Transcription",
        file_name=output_file_name,
        mime="text/plain",
        data=transcription.encode('utf-8')
    )

    st.session_state.processed = True