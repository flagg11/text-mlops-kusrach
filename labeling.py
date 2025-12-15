from rusentilex import load_rusentilex

POS_WORDS, NEG_WORDS = load_rusentilex("data/rusentilex_2017.txt")

def sentiment_label(text):
    pos_count = 0
    neg_count = 0

    for word in text.split():
        if word in POS_WORDS:
            pos_count += 1
        elif word in NEG_WORDS:
            neg_count += 1

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"
