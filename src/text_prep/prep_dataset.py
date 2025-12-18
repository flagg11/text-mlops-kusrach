import pandas as pd
from src.text_prep.lemmatization import lemmatize_text
from src.text_prep.labeling import hybrid_sentiment_label
from src.text_prep.cleaning import clean_vk_mentions, clean_text
from src.utils.config import load_config


config = load_config()

def prep_dataset():
    input_path = config["dataset"]["input_path"]
    output_path = config["dataset"]["output_path"]
    use_lemmatization = config["dataset"].get("use_lemmatization", True)

    df = pd.read_csv(input_path)
    
    df["clean_text"] = df["comment_text"].apply(clean_vk_mentions)
    df["clean_text"] = df["clean_text"].apply(clean_text)


    if use_lemmatization:
        df["processed_text"] = df["clean_text"].apply(lemmatize_text)
    else:
        df["processed_text"] = df["clean_text"]
    
    df["processed_text"] = df["processed_text"].fillna("")

    df = df[df["processed_text"].str.strip() != ""].reset_index(drop=True)

    df["sentiment"] = df["processed_text"].apply(hybrid_sentiment_label)

    df = df[["processed_text", "sentiment"]]
    df.to_csv(output_path, index=False)

    print(f"Dataset prepared: {output_path}")
    print(df["sentiment"].value_counts())

if __name__ == "__main__":
    prep_dataset()
