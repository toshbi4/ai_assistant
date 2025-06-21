import sounddevice as sd
import scipy.io.wavfile as wav
import subprocess
import requests
import tempfile
import os
import json
import re
from TTS.api import TTS  # Coqui TTS

OLLAMA_MODEL = "llama2"  # Or mistral, etc.
WHISPER_MODEL_PATH = "models/ggml-base.en.bin"

# Initialize Coqui TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=True)

def record_audio(filename, duration=5, samplerate=16000):
    print(f"🎤 Recording for {duration} seconds...")
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, samplerate, recording)
    print("✅ Recording complete.")

def transcribe_with_whisper_cpp(audio_path):

    print("🧠 Transcribing with whisper.cpp...")
    cmd = [
        # Path to compiled whisper.cpp binary
        "build/bin/whisper-cli",
        "-m", WHISPER_MODEL_PATH,
        "-f", audio_path,
        "-nt",  # no timestamps
        "-otxt",
        "-of", "../audio2txt"
    ]
    subprocess.run(cmd, cwd="./whisper.cpp/", check=True)
    with open("audio2txt.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

def send_to_ollama(prompt):
    print("📡 Sending to Ollama...")
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": OLLAMA_MODEL,
        "prompt": prompt
    }, stream=True)

    result = ""
    for chunk in response.iter_lines():
        if chunk:
            try:
                data = json.loads(chunk.decode().lstrip("data: "))
                result += data.get("response", "")
            except json.JSONDecodeError:
                continue
    return result

def speak_with_coqui_tts(text):
    print("🗣️ Speaking with Coqui TTS...")

    # Remove emojis and non-basic characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tts.tts_to_file(text=text, file_path=tmp_wav.name)
        samplerate, wav_data = wav.read(tmp_wav.name)

        # Fix: Convert to supported type
        if wav_data.dtype == 'int64':
            wav_data = wav_data.astype('int16')
        elif wav_data.dtype not in ['int16', 'float32']:
            wav_data = wav_data.astype('float32')

        sd.play(wav_data, samplerate)
        sd.wait()

    os.unlink(tmp_wav.name)

def main():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        record_audio(tmp.name)
        transcription = transcribe_with_whisper_cpp(tmp.name)
        print(f"📝 You said: {transcription}")
        response = send_to_ollama(transcription)
        print(f"🤖 Ollama says:\n{response}")
        speak_with_coqui_tts(response)
        os.unlink(tmp.name)

if __name__ == "__main__":
    main()
