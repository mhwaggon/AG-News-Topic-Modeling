Overview

This project uses topic modeling to discover hidden themes in a large collection of news articles. Instead of predicting a label directly, the model looks for word patterns that tend to appear together and groups documents into interpretable topics.

I used the AG News dataset and applied Latent Dirichlet Allocation (LDA) to identify major themes across articles. This makes the project a strong example of practical NLP, unsupervised learning, and interpretable text analysis.

The AG News dataset contains 127,600 articles and is organized into four broad categories: World, Sports, Business, and Sci/Tech.

Project Goals

Load and preprocess a real-world news dataset

Convert article text into a document-term matrix

Train an LDA topic model

Inspect the top words that define each topic

Identify the dominant topic for each article

Review representative articles for each discovered topic

Dataset

This project uses AG News, a widely used news text dataset available through the Hugging Face ecosystem. The dataset includes 120,000 training samples and 7,600 test samples across four classes.

Why this dataset works well for topic modeling:

large enough to produce meaningful topics

clean and easy to load

broad enough to generate distinct themes

useful for comparing discovered topics against known article categories

Tools Used

Python

pandas

nltk

scikit-learn

Hugging Face datasets

The Hugging Face datasets library is designed to make it easy to load and process NLP datasets efficiently.

Method
1. Load the data

The AG News dataset is loaded from Hugging Face and combined into a single DataFrame.

2. Clean the text

Articles are lowercased, punctuation and URLs are removed, and English stopwords are filtered out.

3. Vectorize the text

A CountVectorizer converts the cleaned text into a document-term matrix using unigrams and bigrams.

4. Fit the topic model

An LDA model is trained to learn a fixed number of hidden topics.

5. Interpret the topics

For each topic, the script prints:

top words

label mix by topic

representative documents

6. Export results

The code saves CSV outputs for:

topic keywords

dominant topic per document

representative documents

topic/category counts

Output Files

topic_words.csv

topic_label_counts.csv

representative_documents.csv

document_topics.csv

Example Questions This Project
