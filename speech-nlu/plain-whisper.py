import numpy as np
import speech_recognition as sr
import whisper
import torch
import keyboard

from llama_nlu_v01 import extract_intent 

import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    
    args = parser.parse_args()

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 1000
    recognizer.dynamic_energy_threshold = False

    # microphone
    source = sr.Microphone(sample_rate=16000)

    # whisper model
    model_name = args.model
    if model_name != "large" and not args.non_english:
        model_name += ".en"
    audio_model = whisper.load_model(model_name)

    print(f"✅ Model '{model_name}' loaded.")
    
    with source:
        recognizer.adjust_for_ambient_noise(source)
    
    is_recording = False
    audio_data = None

    print("🎙️ Press [spacebar] to toggle recording ON/OFF. Ctrl+C to exit.\n")

    while True:
        try:
            # wait for space to be pressed
            keyboard.wait("space")

            if not is_recording:
                print("🎤 Recording...")
                with source:
                    audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                audio_data = audio.get_raw_data()
                is_recording = True
            else:
                print("⏹️ Recording stopped. Transcribing...")
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()
                print(text)
                print(extract_intent(text))
                print("\n")
                is_recording = False
            
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break


if __name__ == "__main__":
    main()