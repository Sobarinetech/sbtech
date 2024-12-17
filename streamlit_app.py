import streamlit as st
import whisper
from whisper.tokenizer import get_tokenizer
import torch
import librosa
import numpy as np
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

st.title("Call Recording Transcription with Diarization")

# Load Whisper Model
model = whisper.load_model("base")
tokenizer = get_tokenizer('multilingual')

# Function to process audio file and perform transcription
def transcribe_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    input_values = tokenizer(audio, return_tensors='pt', padding='longest').input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription

# Function to perform diarization
def diarize_audio(file_path):
    # Diarization logic using Whisper-Diarization or other tools
    # Placeholder logic for demonstration
    return ["Speaker 1: Hello, how are you?", "Speaker 2: I'm good, thanks!"]

# Upload file widget
uploaded_file = st.file_uploader("Upload a call recording", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Transcribe and diarize audio
    transcription = transcribe_audio(file_path)
    diarized_text = diarize_audio(file_path)

    # Display transcription
    st.subheader("Transcription")
    st.text(transcription)

    # Display diarized text
    st.subheader("Diarization")
    for line in diarized_text:
        st.text(line)
