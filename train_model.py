import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

# Load CSV dataset
df = pd.read_csv('restaurant_reviews.csv')

# Clean text
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

df['clean_review'] = df['Review'].apply(clean_text)

# Features
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(df['clean_review']).toarray()
y = df['Liked']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))

print("✅ Model trained with CSV dataset!")
