import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud, STOPWORDS
import nltk


nltk.download("stopwords")
from nltk.corpus import stopwords

russian_stopwords = set(stopwords.words("russian"))


os.makedirs("figures", exist_ok=True)

raw_df = pd.read_csv("data/raw/comments.csv")
clean_df = pd.read_csv("data/clean/clean_comments.csv")
labeled_df = pd.read_csv("data/labeled/comments.csv")


counts = pd.DataFrame({
    "stage": ["raw", "cleaned", "labeled"],
    "count": [len(raw_df), len(clean_df), len(labeled_df)]
})
plt.figure(figsize=(6,4))
sns.barplot(x="stage", y="count", data=counts)
plt.title("Количество комментариев на разных этапах")
plt.ylabel("Количество")
plt.xlabel("Этап обработки")
plt.savefig("figures/comments_count_by_stage.png")
plt.show()


def add_length(df, text_col):
    df['length_words'] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    df['length_chars'] = df[text_col].astype(str).apply(len)
    return df

raw_df = add_length(raw_df, "comment_text")
clean_df = add_length(clean_df, "clean_text")
labeled_df = add_length(labeled_df, "lemmatized")

for df, name in zip([raw_df, clean_df, labeled_df], ["raw","cleaned","labeled"]):
    plt.figure(figsize=(8,4))
    sns.histplot(df['length_words'], bins=30, kde=True)
    plt.title(f"Распределение длины комментариев по словам ({name})")
    plt.xlabel("Количество слов")
    plt.ylabel("Количество комментариев")
    plt.savefig(f"figures/length_words_{name}.png")
    plt.show()


plt.figure(figsize=(6,4))
sns.countplot(x="sentiment", data=labeled_df, order=["negative","neutral","positive"])
plt.title("Распределение комментариев по тональности")
plt.xlabel("Тональность")
plt.ylabel("Количество")
plt.savefig("figures/sentiment_distribution.png")
plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(x="sentiment", y="length_words", data=labeled_df, order=["negative","neutral","positive"])
plt.title("Длина комментариев по тональности")
plt.xlabel("Тональность")
plt.ylabel("Количество слов")
plt.savefig("figures/length_by_sentiment.png")
plt.show()


for df, name, col in zip([raw_df, clean_df, labeled_df], ["raw","cleaned","labeled"], ["comment_text","clean_text","lemmatized"]):
    text = " ".join(df[col].astype(str))
    wc = WordCloud(width=800, height=400, background_color="white",
               stopwords=russian_stopwords).generate(text)
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Облако слов ({name})")
    plt.savefig(f"figures/wordcloud_{name}.png")
    plt.show()


for df, name in zip([raw_df, clean_df, labeled_df], ["raw","cleaned","labeled"]):
    plt.figure(figsize=(8,4))
    sns.histplot(df['length_chars'], bins=30, kde=True)
    plt.title(f"Распределение длины комментариев по символам ({name})")
    plt.xlabel("Количество символов")
    plt.ylabel("Количество комментариев")
    plt.savefig(f"figures/length_chars_{name}.png")
    plt.show()

print("EDA графики сохранены в папку figures/")
