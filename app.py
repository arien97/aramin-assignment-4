from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


# Fetch dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups.data

# Initialize TF-IDF Vectorizer
stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
X_tfidf = vectorizer.fit_transform(documents)

# Apply Truncated SVD for dimensionality reduction (Latent Semantic Analysis)
n_components = 110  # Number of components for dimensionality reduction
svd = TruncatedSVD(n_components=n_components)
X_lsa = svd.fit_transform(X_tfidf)

# Function to search for the top 5 most similar documents given a query
def search_engine(query):
    """
    Function to search for top 5 similar documents given a query.
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # Convert query to vector
    query_vec = vectorizer.transform([query])
    query_lsa = svd.transform(query_vec)

    # Compute cosine similarity
    similarities = cosine_similarity(query_lsa, X_lsa)[0]

    # Get top 5 most similar documents
    top_indices = np.argsort(similarities)[::-1][:5]
    top_similarities = similarities[top_indices]
    top_documents = [documents[i] for i in top_indices]

    return top_documents, top_similarities, top_indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities.tolist(), 'indices': indices.tolist()}) 

if __name__ == '__main__':
    app.run(debug=True)