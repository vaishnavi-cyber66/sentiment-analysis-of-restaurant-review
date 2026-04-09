from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (important for Render)
nltk.download('stopwords')

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# Clean function (same as training)
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    review = clean_text(review)

    data = cv.transform([review]).toarray()
    prediction = model.predict(data)[0]

    result = "Positive 😊" if prediction == 1 else "Negative 😠"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run()
