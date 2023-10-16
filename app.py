import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-french"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = librosa.load(path, sr=16_000)
    return speech_array


inputs = processor(speech_file_to_array_fn("path"), sampling_rate=16_000, return_tensors="pt", padding=True)


with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentence = processor.batch_decode(predicted_ids)

tokenizer = AutoTokenizer.from_pretrained('Alwaly/french_sentiment_analysis')
model = BertForSequenceClassification.from_pretrained('Alwaly/french_sentiment_analysis')
token = tokenizer(text=predicted_sentence,
            truncation=True,
            padding='max_length',
            return_tensors='pt')

inputs = token.to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted probabilities and class labels
logits = outputs.logits
predicted_class = logits.argmax().item()

print(predicted_class)