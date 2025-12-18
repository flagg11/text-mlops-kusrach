from src.text_prep.rusentilex import load_rusentilex
from collections import Counter
import re

LEXICON = load_rusentilex("data/rusentilex_2017.txt")

def hybrid_sentiment_label(text, model=None, vectorizer=None):
    tokens = re.findall(r'\b\w+\b', text.lower())
    sentiments = [LEXICON[t] for t in tokens if t in LEXICON]
    counts = Counter(sentiments)

    if counts:
        pos_count = counts.get("positive", 0)
        neg_count = counts.get("negative", 0)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        elif pos_count > 0: 
            return "positive"
        elif neg_count > 0:
            return "negative"
        else:
            return "neutral"

    elif model and vectorizer:
        X = vectorizer.transform([text])
        return model.predict(X)[0]
    else:
        return "neutral"