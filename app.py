import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re
import requests
from bs4 import BeautifulSoup
import json

app = Flask(__name__)

# Data teks dan label sentimen
data = {
    "teks": [
        "mahkamah keluarga",
        "redbull",
        "ANIS RI 1",
        "Anies terbaik",
        "Tetap Anies terbaik",
        "AMIN 2024 RI 1",
        "Anies my next presiden",
        "Indonesia hancur",
        "pilih amin",
        "Tae Prabowo",
        "Prabowo sudah tua",
        "anak walikota",
        "anak presiden",
        "muka songong",
        "tidur aja",
        "pura-pura polos",
        "Cara ngomongnya dan raut wajahnya coba perhatikan",
        "mau muntah mas",
        "muka tidak tau malu",
        "Anies for President",
        "Bismillah Anies",
        "Amin 2024",
        "penculik n penghianatan",
        "sekelas MK aja di atur.apa lagi KPU",
        "planga plongo",
        "mukanya kaya orang penyakitan",
        "Capres abadi",
        "Paling cerdas ya pak anies",
        "siap siap kecewa lg",
        "siap siap kecewa lagi",
        "calon abadi",
        "Anis Baswedan RI 2024",
        "IBU presiden siapa?",
        "Anies solusinya",
        "JANJI NYA AJA GA DI TEPATIN",
        "yakin makmur",
        "MUKA SI GIBRAN KOK NGESELIN YA",
        "mencedrai hukum",
        "Ada jutaan anak muda hebat di Indonesia yg kelasnya jauh di atas Gibran",
        "ga tau malunya itu lho",
        "Tetap anies 2024",
        "HATI HATI ROMBONGAN BANTENG LAGI BERSANDIWARA",
        "Yang gini yang buat indonesia gampang di bodhin",
        "takut karna mulutmu gak bisa dipercaya",
        "Pinokio",
        "All In Ganjar",
        "Ganjar presiden 2024",
        "Ganjar Mahfud Presiden 2024",
        "milih prabowo sama aja milih ganjar",
        "rakyat sudah muak politik dinasti",
        "TAKUT SAMA KETOLOLANNYA",
        "kepedean menang ya",
        "Anies Baswedan pemimpin Indonesia terbaik",
        "sama sama banteng",
        "muka so polos",
        "Takut pas udh jadi presiden menghilang",
        "ga ada kok yg takut",
        "anda aja kegeeran",
        "karena ingusan dan sering manufer harus waspada"
    ],
    "sentimen": [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
}

# Membuat DataFrame dari data
df = pd.DataFrame.from_dict(data)

# Membagi dataset menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(df['teks'], df['sentimen'], test_size=0.2, random_state=42)

# Membangun model TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Melatih model Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Melakukan prediksi pada data pengujian
y_pred = model.predict(X_test_tfidf)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi Model:", accuracy)

# Fungsi untuk membersihkan teks dari karakter khusus dan link
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Menghapus link
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Menghapus karakter khusus kecuali huruf dan angka
    return text

# Fungsi untuk menganalisis sentimen pada teks tertentu
def analyze_sentiment(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return prediction[0]

# Fungsi untuk mendapatkan komentar dari Instagram
def get_instagram_comments(url):
    comments = []
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    scripts = soup.find_all('script', {'type': 'text/javascript'})
    for script in scripts:
        if 'window.__additionalDataLoaded' in str(script):
            data = re.search(r'{.*}', str(script)).group()
            json_data = json.loads(data)
            edges = json_data['graphql']['shortcode_media']['edge_media_to_comment']['edges']
            for edge in edges:
                comment_text = edge['node']['text']
                cleaned_comment = clean_text(comment_text)
                comments.append(cleaned_comment)
    return comments

# Endpoint API untuk menganalisis sentimen pada URL Instagram
@app.route('/analyze_instagram', methods=['POST'])
def analyze_instagram_api():
    data = request.json
    url = data['url']
    comments = get_instagram_comments(url)
    
    positive_comments = 0
    negative_comments = 0

    for comment in comments:
        sentiment = analyze_sentiment(comment)

        if sentiment > 0:
            positive_comments += 1
        elif sentiment < 0:
            negative_comments += 1

    result = {
        'url': url,
        'positive_comments': positive_comments,
        'negative_comments': negative_comments
    }

    return jsonify(result)

# Endpoint API untuk menganalisis sentimen pada teks umum
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment_api():
    data = request.json
    text = data['text']
    sentiment = analyze_sentiment(text)
    result = {'sentiment': sentiment}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
