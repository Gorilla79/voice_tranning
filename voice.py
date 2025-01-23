import torch
from TTS.api import TTS
import os

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to your voice samples
voice_samples_folder = r"D:\capstone\25_01_24"
voice_samples = [
    os.path.join(voice_samples_folder, f"voice ({i}).wav") for i in range(1, 7)
]

# List available TTS models
tts = TTS(model_name=None)  # TTS 객체 생성
print("Available TTS models:", tts.list_models())  # 모델 리스트 확인

# Initialize TTS model (ensure the model name is correct)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)

# Combine multiple voice samples into a single target speaker representation
# This requires selecting one file as the primary speaker sample
target_speaker_wav = voice_samples[0]  # Use the first voice file as the main speaker

# Generate audio from text
output_text = "안녕하세요, 반갑습니다! 이 목소리는 학습된 목소리를 기반으로 생성되었습니다."
output_path = r"D:\capstone\output.wav"

# Generate TTS audio
tts.tts_to_file(
    text=output_text,
    speaker_wav=target_speaker_wav,  # Single speaker file to clone
    language="ko",  # Adjust the language if necessary
    file_path=output_path
)

print(f"TTS audio has been saved to {output_path}")