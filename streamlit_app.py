import streamlit as st
import librosa
import soundfile as sf
import whisper

def transcribe_audio(audio_file):
    model = whisper.load_model("base")  # Correct method for loading Whisper model
    result = model.transcribe(audio_file)
    return result['text']
    
def diarize_audio(text):
    # This is a placeholder for basic diarization. You would need more advanced techniques here.
    speakers = []
    current_speaker = "Speaker 1"  # Start with Speaker 1
    for word in text.split():
        if len(speakers) % 2 == 0:
            current_speaker = "Speaker 1"
        else:
            current_speaker = "Speaker 2"
        speakers.append(current_speaker)
    return speakers

def combine_results(transcription, speakers):
    combined_text = ""
    for word, speaker in zip(transcription.split(), speakers):
        combined_text += f"{speaker}: {word} "
    return combined_text

def main():
    st.title("Call Transcription and Diarization App (Basic)")

    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if audio_file is not None:
        try:
            # Load audio file
            audio, sr = librosa.load(audio_file, sr=None)
            sf.write("temp.wav", audio, sr, subtype='PCM_16')

            # Transcribe audio using Whisper
            transcription = transcribe_audio("temp.wav")

            # Perform basic diarization (this is a placeholder logic)
            speakers = diarize_audio(transcription)

            # Combine results with speaker labels
            combined_text = combine_results(transcription, speakers)

            st.write("Combined Transcription with Speaker Labels:")
            st.write(combined_text)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
