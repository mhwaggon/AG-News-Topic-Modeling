import re
import nltk
import numpy as np
import pandas as pd

from datasets import load_dataset
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

nltk.download("stopwords")

dataset = load_dataset("ag_news")
df = pd.concat(
    [
        pd.DataFrame(dataset["train"]),
        pd.DataFrame(dataset["test"])
    ],
    ignore_index=True
)

label_map = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

df["label_name"] = df["label"].map(label_map)

sample_size = 20000
df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].astype(str).map(clean_text)

vectorizer = CountVectorizer(
    stop_words=list(stop_words),
    max_df=0.90,
    min_df=10,
    ngram_range=(1, 2)
)

dtm = vectorizer.fit_transform(df["clean_text"])

n_topics = 8

lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    learning_method="batch",
    max_iter=10
)

topic_matrix = lda.fit_transform(dtm)

feature_names = np.array(vectorizer.get_feature_names_out())

def get_top_words(model, feature_names, n_top_words=12):
    rows = []
    for topic_idx, topic_weights in enumerate(model.components_):
        top_indices = topic_weights.argsort()[::-1][:n_top_words]
        top_terms = feature_names[top_indices]
        rows.append(
            {
                "topic": topic_idx,
                "top_words": ", ".join(top_terms)
            }
        )
    return pd.DataFrame(rows)

topic_words_df = get_top_words(lda, feature_names, n_top_words=12)

df["dominant_topic"] = topic_matrix.argmax(axis=1)
df["topic_probability"] = topic_matrix.max(axis=1)

topic_label_counts = (
    df.groupby(["dominant_topic", "label_name"])
      .size()
      .reset_index(name="count")
      .sort_values(["dominant_topic", "count"], ascending=[True, False])
)

representative_docs = (
    df.sort_values(["dominant_topic", "topic_probability"], ascending=[True, False])
      .groupby("dominant_topic")
      .head(3)[["dominant_topic", "topic_probability", "label_name", "text"]]
      .reset_index(drop=True)
)

print("\nTOP WORDS PER TOPIC\n")
print(topic_words_df.to_string(index=False))

print("\nLABEL MIX WITHIN EACH DISCOVERED TOPIC\n")
print(topic_label_counts.to_string(index=False))

print("\nREPRESENTATIVE DOCUMENTS\n")
print(representative_docs.to_string(index=False))

topic_words_df.to_csv("topic_words.csv", index=False)
topic_label_counts.to_csv("topic_label_counts.csv", index=False)
representative_docs.to_csv("representative_documents.csv", index=False)
df[["text", "label_name", "dominant_topic", "topic_probability"]].to_csv(
    "document_topics.csv",
    index=False
)
