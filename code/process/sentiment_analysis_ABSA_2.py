import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import numpy as np
import concurrent.futures

tqdm.pandas()

class SentimentAnalysis:
    COLS_KEEP = ['timestamp', 'country', 'is_en', 'trump_sentiment_label', 
                 'biden_sentiment_label', 'mentions_trump', 'mentions_biden', 'likes', 
                 'user_join_date', 'user_followers_count']

    def __init__(self, input_file_path, output_file_path, threshold=0.05, batch_size=32):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.threshold = threshold
        self.batch_size = batch_size

        # Attempt to force use of CUDA with error handling
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device.type == "cuda":
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                print("CUDA not available, using CPU.")
        except Exception as e:
            print(f"Error initializing device: {e}")
            self.device = torch.device("cpu")
            print("Falling back to CPU.")

        # Load the model and tokenizer
        self.absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
        self.absa_model = AutoModelForSequenceClassification.from_pretrained(
            "yangheng/deberta-v3-base-absa-v1.1"
        ).to(self.device)

        # Load the data
        self.df = pd.read_csv(self.input_file_path)

    def process_tweets(self):
        """Calculate sentiment labels for Trump and Biden using parallel batch processing."""
        print("Starting sentiment analysis...")

        self.df = self.df[self.df['is_en'] == True].copy()

        # Initialize a progress bar for the entire dataset
        total_iterations = len(self.df) * 2  # Each row is processed for Trump and Biden
        with tqdm(total=total_iterations, desc="Processing Sentiment", position=0, leave=True) as pbar:
            trump_sentiments = []
            biden_sentiments = []

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []

                # Process the dataset in batches
                for start_idx in range(0, len(self.df), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, len(self.df))
                    batch_text = self.df['textdata'].iloc[start_idx:end_idx]

                    # Submit tasks for both Trump and Biden sentiment processing
                    futures.append(executor.submit(self.get_batch_aspect_sentiment, batch_text, 'trump'))
                    futures.append(executor.submit(self.get_batch_aspect_sentiment, batch_text, 'biden'))

                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()

                    if "trump" in result:
                        trump_sentiments.extend(result['trump'])
                    elif "biden" in result:
                        biden_sentiments.extend(result['biden'])

                    # Update the progress bar
                    pbar.update(self.batch_size)

            # Assign the sentiment labels to the DataFrame
            self.df['trump_sentiment_label'] = trump_sentiments
            self.df['biden_sentiment_label'] = biden_sentiments

            # Keep only relevant columns
            self.df = self.df[self.COLS_KEEP]

            # Save the results to a CSV file
            self.df.to_csv(self.output_file_path, index=False)
            print(f"Sentiment analysis results saved to {self.output_file_path}")


    def get_batch_aspect_sentiment(self, batch_texts, aspect):
        """
        Process sentiment for a batch of texts for a given aspect (Trump or Biden).
        If mentions are 0, return None for those rows.
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
            ).to(self.device)

            with torch.no_grad():
                outputs = self.absa_model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).cpu().numpy()

            labels = ["negative", "neutral", "positive"]
            relevant_sentiments = [labels[np.argmax(prob)] for prob in probs]

            for idx, sentiment in zip(relevant_texts.index, relevant_sentiments):
                sentiment_labels[batch_texts.index.get_loc(idx)] = sentiment

        return {aspect: sentiment_labels}


    def __repr__(self):
        return f'SentimentModel:{self.absa_model}'


if __name__ == "__main__":
    input_file = '../../data/train/processed/processed_data.csv'
    output_file = '../../data/train/processed/ABSA_processed_data.csv'

    analyzer = SentimentAnalysis(input_file, output_file)
    analyzer.process_tweets()

    df = pd.read_csv(output_file)

    biden_tweets = df[df['mentions_trump'] == 1]
    trump_tweets = df[df['mentions_biden'] == 1]

    biden_sentiment_counts = biden_tweets['biden_sentiment_label'].value_counts()
    trump_sentiment_counts = trump_tweets['trump_sentiment_label'].value_counts()

    print("Biden - Sentiment Counts:\n", biden_sentiment_counts)
    print("===================")
    print("Trump - Sentiment Counts:\n", trump_sentiment_counts)
