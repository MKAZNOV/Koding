from flask import Flask, render_template, request
import torch
import re
from transformers import BertForSequenceClassification, AutoTokenizer
import pandas as pd
# Load model dan tokenizer IndoBERT
model_path = "model_sentimen_clean"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Membaca stopwords dari file
stop_words = list(pd.read_csv('https://raw.githubusercontent.com/ShinyQ/E-Wallet-Sentiment-Analysis_IFest-Unpad-2021/main/dataset/processed/stopwords_id_satya.txt', header=None)[0])

# Membaca kamus kata alay dari file
kata_baku = pd.read_csv('https://raw.githubusercontent.com/ShinyQ/One-Click-Sentiment_BE/main/app/dataset/Kamu-Alay.csv')
kata_baku = kata_baku.set_index("kataAlay")["kataBaik"].to_dict()
# Fungsi cleansing
def cleansing(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub('(@\w+|#\w+)', '', text)
    text = re.sub('<.*?>', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub("\n", " ", text)
    text = re.sub('(s{2,})', ' ', text)
    
    temp_text_split = []
    for word in text.split():
        if word not in stop_words:
            if word in kata_baku:
                word = kata_baku[word]
            if len(word) > 3:
                temp_text_split.append(word)
    
    return ' '.join(temp_text_split)

# Fungsi prediksi sentimen
def predict_sentiment(text):
    # Preprocessing
    cleaned_text = cleansing(text)
    
    # Tokenisasi
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    # Prediksi sentimen
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        sentiment_id = torch.argmax(predictions, dim=1).item()
    
    # Label sentimen
    labels = {0: "Negatif", 1: "Netral", 2: "Positif"}
    return labels.get(sentiment_id, "Tidak Diketahui")

# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        text = request.form['text']
        sentiment = predict_sentiment(text)
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
