import torch
import streamlit as st
import io
import os
import subprocess
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, GenerationConfig

st.title("Cantonese ASR Transcription")

if 'processed' not in st.session_state:
    st.session_state.processed = False

uploaded_file = st.file_uploader("Upload your video file", type=["mp4"], key="file_uploader")

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

    uploaded_file = None;  # Clear the uploaded file to free up memory
    # Compress audio
    compressed_audio_io = io.BytesIO()
    audio.export(compressed_audio_io, format='mp3', bitrate='64k')
    compressed_audio_io.seek(0)

    # Compress video using ffmpeg
    # compressed_video_path = "compressed_video.mp4"
    # with open("input_video.mp4", "wb") as f:
    #     f.write(audio_bytes)
    # subprocess.run(["ffmpeg", "-i", "input_video.mp4", "-vcodec", "libx264", "-crf", "28", compressed_video_path])

    with st.spinner('Processing...'):
        # Process entire audio file
        result = pipe(compressed_audio_io.read())
        transcription = result["text"]

    st.success('Done!')
    st.title("Result")
    with st.expander("Transcription"):
        st.info(transcription)

    # Create download button for transcription
    filename_without_extension = os.path.splitext(uploaded_file.name)[0]
    output_file_name = f"{filename_without_extension}_transcription.txt"
    st.download_button(
        label="Download Transcription",
        file_name=output_file_name,
        mime="text/plain",
        data=transcription.encode('utf-8')
    )

    # Create download button for compressed video
    with open(compressed_video_path, "rb") as f:
        st.download_button(
            label="Download Compressed Video",
            file_name=compressed_video_path,
            mime="video/mp4",
            data=f.read()
        )

    st.session_state.processed = True