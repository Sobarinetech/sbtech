import streamlit as st
import librosa
import soundfile as sf
import whisper


# Function to transcribe audio
def transcribe_audio(audio_file):
  model = whisper.load_model("base")  # Choose a model based on your needs
  result = model.transcribe(audio_file)
  return result['text']

# Function to perform basic diarization (speaker change detection)
def diarize_audio(text):
  speakers = []
  current_speaker = None
  for word in text.split():
    if model.language_model.decode(word)["probs"][0] < 0.5:  # Threshold for speaker change
      current_speaker = f"Speaker {len(speakers)+1}"
    speakers.append(current_speaker)
  return speakers

# Function to combine transcription and speaker labels
def combine_results(transcription, speakers):
  combined_text = ""
  for word, speaker in zip(transcription.split(), speakers):
    combined_text += f"{speaker}: {word} "
  return combined_text

# Streamlit app
def main():
  st.title("Call Transcription and Diarization App (Basic)")

  # File upload
  audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

  if audio_file is not None:
    # Load audio file
    audio, sr = librosa.load(audio_file, sr=None)
    sf.write("temp.wav", audio, sr, subtype='PCM_16')

    # Transcribe audio
    transcription = transcribe_audio("temp.wav")

    # Perform basic diarization
    speakers = diarize_audio(transcription)

    # Combine results
    combined_text = combine_results(transcription, speakers)

    st.write("Combined Transcription with Speaker Labels:")
    st.write(combined_text)

if __name__ == "__main__":
  main()
