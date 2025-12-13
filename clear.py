import pandas as pd
import re
import demoji
import re

df = pd.read_csv("comments.csv")

def clean_vk_mentions(text):
    text = re.sub(r"\[id\d+\|([^\]]+)\]", r"\1", text)
    return text

df['clean_text'] = df['comment_text'].apply(clean_vk_mentions)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text) 
    text = demoji.replace(text, "")   
    text = re.sub(r"[^а-яa-z\s]", "", text) 
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['clean_text'] = df['clean_text'].apply(clean_text)

df = df[df['clean_text'].str.len() > 10].reset_index(drop=True)

print(df.head())
