from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
cv = pickle.load(open('vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    data = cv.transform([review])
    prediction = model.predict(data)[0]

    result = "Positive 😊" if prediction == 1 else "Negative 😠"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run()
