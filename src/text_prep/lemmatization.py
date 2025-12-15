import pymorphy3

morph = pymorphy3.MorphAnalyzer()

def lemmatize_text(text: str) -> str:
    words = text.split()
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmas)
