from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import newspaper

app = Flask(__name__)

# Load the model and tokenizer
model = load_model('model.h5')
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Preprocess function
def preprocess_text(text):
    processed_text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    return processed_text

# Scrape news content from URL using newspaper3k
def scrape_news(url):
    article = newspaper.Article(url)
    article.download()
    article.parse()
    return article.text

# Route to render the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_type = request.form.get('inputType')
        url = request.form.get('url')
        news_text = request.form.get('news_text')

        if input_type == 'url':
            if not url:
                raise ValueError("Please provide a URL.")
            # Scrape news content from the URL
            news_text = scrape_news(url)
        elif input_type == 'text':
            if not news_text:
                raise ValueError("Please provide News Text.")
        
        # Preprocess the input news content
        processed_text = preprocess_text(news_text)

        # Tokenize and pad the input text
        sequence = tokenizer.texts_to_sequences([processed_text])
        data = pad_sequences(sequence, maxlen=500) 

        # Make the prediction
        prediction = model.predict(data)
        result = "Real" if prediction[0][0] > 0.5 else "Fake"

        # Return the result and scraped news text to the frontend
        return render_template('index.html', result=result, url=url, news_text=news_text, selected_input=input_type)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
