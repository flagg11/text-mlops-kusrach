from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def create_tfidf_lr():
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    model = LogisticRegression(max_iter=500)
    return vectorizer, model

def create_count_nb():
    vectorizer = CountVectorizer(max_features=5000, ngram_range=(1,2))
    model = MultinomialNB()
    return vectorizer, model
