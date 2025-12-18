import pandas as pd
import re
import demoji
import re

def clean_vk_mentions(text):
    text = re.sub(r"\[id\d+\|([^\]]+)\]", r"\1", text)
    return text

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text) 
    text = demoji.replace(text, "")   
    text = re.sub(r"[^а-яa-z\s]", "", text) 
    text = re.sub(r"\s+", " ", text).strip()
    return text
