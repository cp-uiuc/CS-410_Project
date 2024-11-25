from typing import Counter
import pandas as pd
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
import numpy as np
import time

tqdm.pandas()

class SentimentAnalysis:

    COLS_KEEP = ['timestamp', 'country', 'candidate', 'is_en', 'trump_sentiment_label', 'biden_sentiment_label', 'mentions_trump', 'mentions_biden', 'likes', 'user_join_date', 'user_followers_count']

    def __init__(self, input_file_path, output_file_path, threshold=0.05, batch_size=16):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.threshold = threshold
        self.batch_size = batch_size

        self.absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
        self.absa_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
        
        # Load the data
        self.df = pd.read_csv(self.input_file_path)


    def process_tweets(self):
        """Process sentiment labels for tweets in batches."""
        print("Starting sentiment analysis...")

        # Add columns for storing sentiment results
        self.df['trump_sentiment_label'] = None
        self.df['biden_sentiment_label'] = None

        with tqdm(total=len(self.df), desc="Processing Sentiment") as pbar:
            for start_idx in range(0, len(self.df), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(self.df))

                batch_texts = self.df['textdata'].iloc[start_idx:end_idx]
                trump_sentiments = self.get_batch_aspect_sentiment(batch_texts, 'trump')
                biden_sentiments = self.get_batch_aspect_sentiment(batch_texts, 'biden')

                self.df.loc[start_idx:end_idx - 1, 'trump_sentiment_label'] = trump_sentiments
                self.df.loc[start_idx:end_idx - 1, 'biden_sentiment_label'] = biden_sentiments

                pbar.update(end_idx - start_idx)

        # Retain only required columns and save results
        self.df = self.df[self.COLS_KEEP]
        self.df.to_csv(self.output_file_path, index=False)
        print(f"Sentiment analysis results saved to {self.output_file_path}")

    def get_batch_aspect_sentiment(self, batch_texts, aspect):
        """
        Process a batch of texts for a specific aspect (Trump or Biden).
        Only analyze rows where the corresponding mentions column equals 1.
        """
        mentions_col = f"mentions_{aspect}"
        mentions_mask = self.df.loc[batch_texts.index, mentions_col] == 1

        sentiment_labels = [None] * len(batch_texts)

        if mentions_mask.any():
            relevant_texts = batch_texts[mentions_mask]

            inputs = self.absa_tokenizer(
                [f"[CLS] {text} [SEP] {aspect} [SEP]" for text in relevant_texts],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.absa_model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).numpy()

            labels = ["negative", "neutral", "positive"]
            relevant_sentiments = [labels[np.argmax(prob)] for prob in probs]

            for idx, sentiment in zip(relevant_texts.index, relevant_sentiments):
                sentiment_labels[batch_texts.index.get_loc(idx)] = sentiment

        return sentiment_labels

    def __repr__(self):
        return f'SentimentModel:{self.absa_model.name_or_path}'

    
    def __repr__(self):
        return f'SentimentModel:{self.model}'


if __name__ == "__main__":
    input_file = '../../data/train/processed/processed_data.csv'
    output_file = '../../data/train/processed/ABSA_processed_data.csv'

    # Initialize with desired model ('VADER' or 'TextBlob')
    analyzer = SentimentAnalysis(input_file, output_file)
    analyzer.process_tweets()

    df = pd.read_csv(output_file)

    biden_tweets = df[df['mentions_trump'] == 1]
    trump_tweets = df[df['mentions_biden'] == 1]

    biden_sentiment_counts = biden_tweets['biden_sentiment_label'].value_counts()

    print("Biden - Sentiment Counts:\n", biden_sentiment_counts)
    print("===================")

    trump_sentiment_counts = trump_tweets['trump_sentiment_label'].value_counts()

    print("Trump - Sentiment Counts:\n", trump_sentiment_counts)



