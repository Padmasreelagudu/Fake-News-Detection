# app.py
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split  # Add this line
import preprocessing

app = Flask(__name__, template_folder='templates')

# Load your dataset
df = pd.read_csv('train.csv')

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['Statement'], df['Label'], test_size=0.2, random_state=42
)

# Text Vectorization
vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore', stop_words='english')
X_train = vectorizer.fit_transform(train_data.apply(preprocessing.preprocess))

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, train_labels)

# Flask route to handle form submission and display predictions
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        article = request.form['article']
        processed_article = preprocessing.preprocess(article)
        vectorized_article = vectorizer.transform([processed_article])
        prediction = clf.predict(vectorized_article)[0]
        return render_template('predictions.html', prediction=prediction, article=article)

    return render_template('page.html')

if __name__ == '__main__':
    app.run(debug=True)
