import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import keyboard

from llama-nlu-v01 import extract_intent 
from sys import platform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name. Use 'list' to show devices.", type=str)
    args = parser.parse_args()

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 1000
    recognizer.dynamic_energy_threshold = False

    # Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphones:")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"{index}: {name}")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Whisper model
    model_name = args.model
    if model_name != "large" and not args.non_english:
        model_name += ".en"
    audio_model = whisper.load_model(model_name)

    print(f"✅ Model '{model_name}' loaded.")
    print("🎙️ Hold [spacebar] to record. Release to transcribe. Ctrl+C to exit.\n")

    with source:
        recognizer.adjust_for_ambient_noise(source)

    transcription = []

    while True:
        try:
            # Spacebar to execute
            keyboard.wait("space")
            print("🎤 Recording... (release to stop)")
            with source:
                audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)

            print("⏹️ Recording stopped. Transcribing...")

            audio_data = audio.get_raw_data()
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
            text = result['text'].strip()
            '''transcription.append(text)

            os.system('cls' if os.name == 'nt' else 'clear')
            print("📝 Transcription:\n")
            for line in transcription:
                print(line)'''
            print(text)
            print(extract_intent(text))
            print("\n")
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break

if __name__ == "__main__":
    main()

