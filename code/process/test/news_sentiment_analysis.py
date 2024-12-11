import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
import torch.nn.functional as F
import numpy as np




tqdm.pandas()

# Download VADER lexicon
nltk.download('vader_lexicon')

MAIN_COLS_KEEP = ['timestamp', "trump_sentiment_label", "mentions_trump", "harris_sentiment_label", "mentions_harris"]

class NewsSentimentAnalysis:

    COLS_KEEP = ['timestamp', "trump_sentiment_label", "mentions_trump", "harris_sentiment_label", "mentions_harris"]

    
    def __init__(self, input_file_path, output_file_path, model = "VADER", threshold=0.05, batch_size=16):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.model = model
        self.threshold = threshold
        
        with open(input_file_path, 'r') as f:
            data = json.load(f)
        self.df = pd.DataFrame(data[0], columns=['textdata', 'timestamp', 'mentions_trump', 'mentions_harris'])
        
        self.COLS_KEEP = MAIN_COLS_KEEP if self.model == 'ABSA' else MAIN_COLS_KEEP + ['harris_sentiment_score', 'trump_sentiment_score']

        if self.model == "VADER":
            self.analyzer = SentimentIntensityAnalyzer()
        elif self.model == "TextBlob":
            self.analyzer = None
        elif self.model == "ABSA":
            self.batch_size = batch_size
            self.absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
            self.absa_model = AutoModelForSequenceClassification.from_pretrained(
                "yangheng/deberta-v3-base-absa-v1.1"
            )
        else: 
            raise ValueError("Invalid model choice. Choose 'VADER' or 'TextBlob' or 'ABSA'.")

    
    def process_news(self):

        if self.model != "ABSA":
            print("Calculating polarity scores and labeling sentiments for each news title...")

            def calculate_sentiment(row):
                """
                Calculate sentiment scores and labels based on mentions in the row.
                """
                trump_score, trump_label = 0, 'neutral'
                harris_score, harris_label = 0, 'neutral'

                # Calculate sentiment for Trump mentions
                if row['mentions_trump'] == 1:
                    trump_score = self.calculate_sentiment_score(row['textdata'])
                    trump_label = self.label_sentiment(trump_score)
                elif row['mentions_trump'] == 0:
                    trump_score = 0
                    trump_label = 'neutral'

                # Calculate sentiment for Harris mentions
                if row['mentions_harris'] == 1:
                    harris_score = self.calculate_sentiment_score(row['textdata'])
                    harris_label = self.label_sentiment(harris_score)
                elif row['mentions_harris'] == 0:
                    harris_score = 0
                    harris_label = 'neutral'

                return trump_score, trump_label, harris_score, harris_label

            # Apply sentiment analysis row-wise
            tqdm.pandas(desc="Processing sentiment analysis")
            results = self.df.progress_apply(calculate_sentiment, axis=1)

            # Unpack results into new columns
            self.df[['trump_sentiment_score', 'trump_sentiment_label',
                    'harris_sentiment_score', 'harris_sentiment_label']] = pd.DataFrame(results.tolist(), index=self.df.index)

            # Keep only relevant columns
            self.df = self.df[self.COLS_KEEP]

            # Save the updated DataFrame to a CSV file
            self.df.to_csv(self.output_file_path, index=False)
            print(f"Sentiment analysis completed. Results saved to {self.output_file_path}.")
        else:
            # Initialize a single progress bar
            with tqdm(total=len(self.df), desc="Processing Sentiment", position=0, leave=True) as pbar:

                # Create lists to store sentiment labels for both Trump and Harris
                trump_sentiments = []
                harris_sentiments = []

                # Process data in batches
                for start_idx in range(0, len(self.df), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, len(self.df))

                    batch_text = self.df['textdata'].iloc[start_idx:end_idx]

                    # Get batch sentiment for Trump and Harris
                    trump_sentiment_batch = self.get_batch_aspect_sentiment(batch_text, 'trump')
                    harris_sentiment_batch = self.get_batch_aspect_sentiment(batch_text, 'harris')

                    # Append the batch results to their respective lists
                    trump_sentiments.extend(trump_sentiment_batch)
                    harris_sentiments.extend(harris_sentiment_batch)

                    # Update the progress bar for the batch
                    pbar.update(end_idx - start_idx)

            # Assign the sentiment labels to their respective columns
            self.df['trump_sentiment_label'] = trump_sentiments
            self.df['harris_sentiment_label'] = harris_sentiments

            # Keep only relevant columns
            self.df = self.df[self.COLS_KEEP]

            # Save the results to a CSV file
            self.df.to_csv(self.output_file_path, index=False)
            print(f"Sentiment analysis results saved to {self.output_file_path}")

    def calculate_sentiment_score(self, text):
        #Calculate sentiment score based on the chosen model
        if self.model == "VADER":
            return self.analyzer.polarity_scores(text)['compound']
        elif self.model == "TextBlob":
            return TextBlob(text).sentiment.polarity

    def label_sentiment(self, score):
        #Labels sentiment based on the score and threshold
        if score >= self.threshold:
            return "positive"
        elif score <= -self.threshold:
            return "negative"
        else:
            return "neutral"
        

    def get_batch_aspect_sentiment(self, batch_texts, aspect):
        """
        Process sentiment for a batch of texts for a given aspect (Trump or Harris).
        If mentions are 0, return None for those rows.
        """
        # Determine the column to check for mentions
        mentions_col = f"mentions_{aspect}"
        
        # Identify rows where mentions are 1
        mentions_mask = self.df.loc[batch_texts.index, mentions_col] == 1

        # Create a placeholder for sentiment labels
        sentiment_labels = [None] * len(batch_texts)

        # Process only rows with mentions
        if mentions_mask.any():
            # Select texts for rows with mentions
            relevant_texts = batch_texts[mentions_mask]
            
            # Tokenize and process sentiment analysis for relevant texts
            inputs = self.absa_tokenizer(
                [f"[CLS] {text} [SEP] {aspect} [SEP]" for text in relevant_texts],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            outputs = self.absa_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).detach().numpy()

            labels = ["negative", "neutral", "positive"]
            # Get sentiment labels for relevant rows
            relevant_sentiments = [labels[np.argmax(prob)] for prob in probs]

            # Map back to the original indices
            for idx, sentiment in zip(relevant_texts.index, relevant_sentiments):
                sentiment_labels[batch_texts.index.get_loc(idx)] = sentiment

        return sentiment_labels

if __name__ == "__main__":
    input_file = '../../../data/test/processed/processed_election_news.json'
    output_file = '../../../data/test/processed/TextBlob_processed_news_data.csv'

    # Initialize with desired model ('VADER' or 'TextBlob')
    analyzer = NewsSentimentAnalysis(input_file, output_file, model = "VADER")
    analyzer.process_news()

    df = pd.read_csv(output_file)

    harris_news = df[df['mentions_harris'] == 1]
    trump_news = df[df['mentions_trump'] == 1]

    if analyzer.model != 'ABSA':
        harris_sentiment_avg = harris_news['harris_sentiment_score'].mean()
        print("Harris - Average Sentiment Score:", harris_sentiment_avg)

    harris_sentiment_counts = harris_news['harris_sentiment_label'].value_counts()

    print("Harris - Sentiment Counts:\n", harris_sentiment_counts)
    print("===================")

    if analyzer.model != 'ABSA':
        trump_sentiment_avg = trump_news['trump_sentiment_score'].mean()
        print("Trump - Average Sentiment Score:", trump_sentiment_avg)

    trump_sentiment_counts = trump_news['trump_sentiment_label'].value_counts()

    print("Trump - Sentiment Counts:\n", trump_sentiment_counts)

