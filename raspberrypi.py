import torch
from TTS.api import TTS

# Function to convert a 3-digit number into Korean text
def number_to_korean(number):
    units = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
    tens = ["", "십", "백"]
    
    result = ""
    num_str = str(number)
    length = len(num_str)

    for idx, digit in enumerate(num_str):
        digit = int(digit)
        if digit > 0:
            result += units[digit]
            if idx < length - 1:  # Add positional suffix (백, 십)
                result += tens[length - idx - 1]
    return result

# Main script
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the TTS model
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)

    # Prompt user for a 3-digit number
    while True:
        user_input = input("Enter a 3-digit number: ")
        if user_input.isdigit() and len(user_input) == 3:
            break
        else:
            print("Invalid input. Please enter exactly 3 digits.")

    # Convert number to Korean text
    korean_number = number_to_korean(int(user_input))
    final_text = f"{korean_number}번 탈락"

    # Define the output path
    output_path = "/home/pi/output.wav"

    # Generate TTS audio
    tts.tts_to_file(
        text=final_text,
        speaker_wav=None,  # Optional: provide a speaker_wav file if voice cloning is needed
        language="ko",
        file_path=output_path
    )

    # Notify the user
    print(f"Audio has been saved to {output_path}. The text is: {final_text}")
