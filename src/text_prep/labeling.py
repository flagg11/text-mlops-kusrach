from .rusentilex import load_rusentilex
from collections import Counter

LEXICON = load_rusentilex("data/rusentilex_2017.txt")

def hybrid_sentiment_label(text, model=None, vectorizer=None):
    tokens = text.lower().split()
    sentiments = [LEXICON[t] for t in tokens if t in LEXICON]
    counts = Counter(sentiments)

    if counts:
        if counts["positive"] > counts["negative"]:
            return "positive"
        elif counts["negative"] > counts["positive"]:
            return "negative"
        else:
            return "neutral"
    elif model and vectorizer:
        X = vectorizer.transform([text])
        return model.predict(X)[0]
    else:
        return "neutral"