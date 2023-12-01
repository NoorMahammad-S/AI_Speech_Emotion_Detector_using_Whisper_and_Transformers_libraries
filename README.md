# AI_Speech_Emotion_Detector_using_Whisper_and_Transformers_libraries
This project demonstrates a simple pipeline for Speech Emotion Recognition using Whisper for speech recognition, RoBERTa for text representation extraction, and BART for emotion detection.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required dependencies.

```bash
pip install -r requirements.txt
```

## Usage

1. Replace `"path_to_your_audio_file.wav"` in the `main()` function of `main.py` with the path to your actual audio file.

2. Run the main:

```bash
python main.py
```

The script will process the audio file, recognize speech using Whisper, extract text representation with RoBERTa, and detect emotion with BART.

## Configuration

- `main.py`: Main script that integrates Whisper, RoBERTa, and BART for speech emotion recognition.

## Models Used

- RoBERTa: [roberta-base](https://huggingface.co/roberta-base)
- BART: [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)
- Whisper: [whisper-large](https://huggingface.co/whisper-large)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

