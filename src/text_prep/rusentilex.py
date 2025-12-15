def load_rusentilex(path):
    lex = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("!") or not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            word = parts[2]
            sentiment = parts[3]
            lex[word] = sentiment
    return lex
