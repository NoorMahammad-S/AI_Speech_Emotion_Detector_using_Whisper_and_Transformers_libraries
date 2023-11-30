from transformers import RobertaTokenizer, RobertaForSequenceClassification, BartTokenizer, BartForSequenceClassification
from whisper.denoiser import Denoiser
import torchaudio
import torch

# Load RoBERTa model
tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
model_roberta = RobertaForSequenceClassification.from_pretrained("roberta-base")

# Load BART model for emotion detection
emotion_model = BartForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
emotion_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli")

# Load Whisper denoiser
denoiser = Denoiser("whisper-large")

def recognize_speech(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    denoised_waveform = denoiser(waveform)
    transcript = " ".join([" ".join(hypo.tokens) for hypo in recognizer.transcribe(denoised_waveform[0].numpy())])
    return transcript

def extract_text_representation(transcript):
    inputs = tokenizer_roberta(transcript, return_tensors="pt")
    outputs = model_roberta(**inputs)
    text_representation = outputs.logits
    return text_representation

def detect_emotion(text_representation):
    inputs = emotion_tokenizer(text_representation, return_tensors="pt")
    outputs = emotion_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map the predicted class to an emotion label (modify as needed)
    emotion_labels = ["happy", "sad", "angry", "neutral"]
    predicted_emotion = emotion_labels[predicted_class]

    return predicted_emotion

def main():
    # Replace with the path to your audio file
    audio_file_path = "path_to_your_audio_file.wav"

    # Speech recognition with Whisper
    transcript = recognize_speech(audio_file_path)

    # Extract text representation using RoBERTa
    text_representation = extract_text_representation(transcript)

    # Emotion detection using BART
    predicted_emotion = detect_emotion(text_representation)
    print("Predicted Emotion:", predicted_emotion)

if __name__ == "__main__":
    main()
