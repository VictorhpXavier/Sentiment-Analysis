# app.py
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model and vectorizer
vectorizer = joblib.load('Dataset/vectorizer.pkl')
model = joblib.load('Dataset/model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    print(text)
    
    # Vectorize the input text
    text_vec = vectorizer.transform([text])
    
    # Predict the sentiment
    prediction = model.predict(text_vec)
    
    if prediction == 1:
        result = "This is a positive text."
    elif prediction == -1:
        result = "This is a negative text."
    else:
        result = "This is a neutral text."

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
