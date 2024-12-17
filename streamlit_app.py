import streamlit as st
import librosa
import soundfile as sf
import whisper
from pyannote.audio import Pipeline
import numpy as np

# Initialize Whisper model for transcription
def transcribe_audio(audio_file):
    model = whisper.load_model("base")  # Load Whisper model
    result = model.transcribe(audio_file)
    return result['text']

# Initialize Pyannote pipeline for speaker diarization
def diarize_audio(audio_file):
    # Initialize the diarization pipeline from Pyannote
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

    # Apply diarization to the audio file
    diarization = pipeline({'uri': 'filename', 'audio': audio_file})
    
    # Extract the speaker segments and map to speaker labels
    speakers = []
    for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append(speaker)
    
    return speakers

def combine_results(transcription, speakers):
    combined_text = ""
    words = transcription.split()
    speaker_idx = 0
    for word in words:
        if speaker_idx < len(speakers):
            combined_text += f"{speakers[speaker_idx]}: {word} "
            speaker_idx += 1
        else:
            combined_text += f"Speaker {speaker_idx + 1}: {word} "
    return combined_text

def main():
    st.title("Call Transcription and Diarization App (with Pyannote)")

    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if audio_file is not None:
        try:
            # Load audio file
            audio, sr = librosa.load(audio_file, sr=None)
            sf.write("temp.wav", audio, sr, subtype='PCM_16')

            # Transcribe audio using Whisper
            transcription = transcribe_audio("temp.wav")

            # Perform speaker diarization using Pyannote
            speakers = diarize_audio("temp.wav")

            # Combine results with speaker labels
            combined_text = combine_results(transcription, speakers)

            # Display the combined text with speaker labels
            st.write("Combined Transcription with Speaker Labels:")
            st.write(combined_text)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
