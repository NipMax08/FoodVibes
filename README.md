# FoodVibes
# Sentiment Analysis of Food Reviews

This project delves into the concept of **Sentiment Analysis** within the context of food reviews. Sentiment Analysis involves examining the sentiment of textual content to classify opinions into categories such as Positive, Neutral, or Negative. The primary objective is to apply **Natural Language Processing (NLP)** techniques to analyze and classify sentiments in recipe reviews, uncovering trends and insights.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
3. [Sentiment Labeling](#sentiment-labeling)
4. [Model Training](#model-training)
5. [Conclusion](#conclusion)
6. [Future Work](#future-work)
7. [References](#references)

---

## Introduction

### What is NLP and Sentiment Analysis?

**Natural Language Processing (NLP)** is a field of artificial intelligence focusing on analyzing and processing human language. It encompasses various tasks such as machine translation, spam detection, summarization, and sentiment analysis.

In this project, we focus on **Sentiment Analysis**, which involves determining the emotional tone behind reviews to classify them into three categories:
- Positive
- Neutral
- Negative

NLP techniques enable us to analyze customer feedback efficiently, providing actionable insights for data-driven decision-making.

---

## Data Collection and Preprocessing

### Web Scraping

The dataset was scraped from **Food.com** using **Selenium**, a robust automation tool for extracting dynamic content from websites. Selenium's ability to handle JavaScript-based web pages made it the ideal choice for this project.

### Text Preprocessing

Text preprocessing is essential to clean and prepare the data for analysis. Key steps include:
- Removing punctuation, URLs, and stop words
- Lowercasing text
- Tokenization
- Handling emoticons, emojis, and slang
- Lemmatization and stemming
- Spell checking with TextBlob

These steps ensured a structured and ready dataset for downstream tasks. The cleaned data was exported to a CSV file for further processing.

---

## Sentiment Labeling

The **cardiffnlp/twitter-roberta-base-sentiment** model from Hugging Face was used for sentiment labeling. This pre-trained model, designed for Twitter data, efficiently classified reviews into sentiment categories. Steps included:
1. Tokenization and text chunking for model input compatibility.
2. Applying the sentiment model to each review.
3. Mapping model outputs (`label_0`, `label_1`, `label_2`) to categories (Negative, Neutral, Positive).

The resulting dataset included additional columns for sentiment labels and confidence scores.

---

## Model Training

Various approaches were tested for sentiment classification:
1. **Word Embeddings**: GloVe and Word2Vec
2. **TF-IDF** and **Bag of Words** representations
3. **Transformers**: Brief exploration of advanced models like GRU with GloVe embeddings.

Each method highlighted the challenges of handling textual data and imbalanced datasets. Results were evaluated using metrics like accuracy and F1-score.

---

## Conclusion

This project provided valuable insights into NLP techniques for sentiment analysis, particularly in the context of imbalanced datasets. While some models performed well, limitations in computational resources and dataset complexity impacted overall performance.

**Best Performing Model**: GRU with GloVe embeddings showed the most promise, effectively capturing nuanced sentiments.

---
## Current Work

- Implment the model with more data.


## Future Work

- Explore ensemble methods for improved classification.
- Fine-tune hyperparameters for better optimization.
- Integrate deep learning models like **BERT** for enhanced sentiment analysis.
- Address class imbalance with advanced sampling techniques.


