from rusentilex import load_rusentilex
from collections import Counter

LEXICON = load_rusentilex("data/rusentilex_2017.txt")

def sentiment_label(text):
    tokens = text.split()
    sentiments = [LEXICON[t] for t in tokens if t in LEXICON]

    counts = Counter(sentiments)

    if counts["positive"] > counts["negative"]:
        return "positive"
    elif counts["negative"] > counts["positive"]:
        return "negative"
    else:
        return "neutral"
