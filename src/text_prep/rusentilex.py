def load_rusentilex(path):
    positive = set()
    negative = set()

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("!"):
                continue

            parts = [p.strip() for p in line.split(",")]

            if len(parts) < 4:
                continue  

            word = parts[0].lower()
            sentiment = parts[3].lower()

            if sentiment == "positive":
                positive.add(word)
            elif sentiment == "negative":
                negative.add(word)

    return positive, negative
