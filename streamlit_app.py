import streamlit as st
import speechbrain as sb
from speechbrain.pretrained import diarization
from speechbrain.pretrained import whisper
import torchaudio
import torch
import os

# Check for GPU availability
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.empty_cache()  # Clear cache if using GPU
else:
    device = "cpu"

# Initialize models outside the main function for efficiency
try:
    diar = diarization.SpeakerDiarization.from_hparams(
        source="speechbrain/diarization-xvector-voxceleb",
        savedir="pretrained_models/diarization",
    ).to(device)
    asr_model = whisper.load_whisper("base").to(device)
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure you have internet connectivity for initial download.")
    st.stop()



def process_audio(audio_path):
    try:
        # Diarization
        with torch.no_grad():
            diar.to(device)
            _, _, speaker_times = diar.process_file(audio_path)

        # ASR (Whisper)
        with torch.no_grad():

            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = waveform.to(device)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            transcription = asr_model.transcribe(waveform)
            segments = transcription["segments"]

        # Align and combine
        diarized_transcript = []
        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            speaker = "Unknown"
            for spk, times in speaker_times.items():
                for start, end in times:
                    if start <= start_time <= end or start <= end_time <= end:
                        speaker = spk
                        break
                else:
                    continue
                break
            diarized_transcript.append(f"[{speaker} {start_time:.2f}-{end_time:.2f}]: {text}")

        return diarized_transcript

    except RuntimeError as e:
        if "Expected more than 1 value per channel" in str(e):
            return ["Error: Audio file is likely mono. Diarization requires stereo audio."]
        else:
            return [f"A RuntimeError occurred during processing: {e}"]
    except Exception as e:
        return [f"An unexpected error occurred: {e}"]

st.title("Call Recording Transcription with Diarization")

uploaded_file = st.file_uploader("Upload a call recording (WAV or MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file to temporary location
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    audio_path = "temp_audio.wav"

    st.audio(uploaded_file, format=uploaded_file.type)

    if st.button("Transcribe and Diarize"):
        with st.spinner("Processing audio..."):
            transcript = process_audio(audio_path)

        st.subheader("Diarized Transcript:")
        for line in transcript:
            st.write(line)

    # Clean up temporary file
    os.remove("temp_audio.wav")
