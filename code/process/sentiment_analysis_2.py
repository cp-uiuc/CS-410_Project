import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import numpy as np
import concurrent.futures

# Download VADER lexicon
import nltk
import ssl

tqdm.pandas()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('vader_lexicon')

MAIN_COLS_KEEP = ['timestamp', 'country', 'is_en', 'trump_sentiment_label', 
                 'biden_sentiment_label', 'mentions_trump', 'mentions_biden', 'likes', 
                 'user_join_date', 'user_followers_count']
class SentimentAnalysis:
    def __init__(self, input_file_path, output_file_path, model = "VADER", threshold=0.05):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.model = model
        self.threshold = threshold
        self.df = pd.read_csv(self.input_file_path)

        self.COLS_KEEP = MAIN_COLS_KEEP if self.model == 'ABSA' else MAIN_COLS_KEEP + ['biden_sentiment_score', 'trump_sentiment_score']

        if self.model == "VADER":
            self.analyzer = SentimentIntensityAnalyzer()
        elif self.model == "TextBlob":
            self.analyzer = None
        elif self.model == 'ABSA':
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
        else:
            raise ValueError("Invalid model choice. Choose 'VADER' or 'TextBlob'.")
        
        #Load the data
        self.df = pd.read_csv(self.input_file_path)

    def process_tweets(self):
        # Filter English tweets
        self.df = self.df[self.df['is_en'] == True].copy()

        if self.model != 'ABSA':
            print("Calculating polarity scores and labeling sentiments for each tweet...")

            def calculate_sentiment(row):
                """
                Calculate sentiment scores and labels based on mentions in the row.
                """
                trump_score, trump_label = None, None
                biden_score, biden_label = None, None

                # Calculate sentiment for Trump mentions
                if row['mentions_trump'] == 1:
                    trump_score = self.calculate_sentiment_score(row['textdata'])
                    trump_label = self.label_sentiment(trump_score)

                # Calculate sentiment for Biden mentions
                if row['mentions_biden'] == 1:
                    biden_score = self.calculate_sentiment_score(row['textdata'])
                    biden_label = self.label_sentiment(biden_score)

                return trump_score, trump_label, biden_score, biden_label

            # Apply sentiment analysis row-wise
            tqdm.pandas(desc="Processing sentiment analysis")
            results = self.df.progress_apply(calculate_sentiment, axis=1)

            # Unpack results into new columns
            self.df[['trump_sentiment_score', 'trump_sentiment_label',
                    'biden_sentiment_score', 'biden_sentiment_label']] = pd.DataFrame(results.tolist(), index=self.df.index)

            # Keep only relevant columns
            self.df = self.df[self.COLS_KEEP]

            # Save the updated DataFrame to a CSV file
            self.df.to_csv(self.output_file_path, index=False)
            print(f"Sentiment analysis completed. Results saved to {self.output_file_path}.")
        else:
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

    def calculate_sentiment_score(self, text):
        #Calculate sentiment score based on the chosen model
        if self.model == "VADER":
            return self.analyzer.polarity_scores(text)['compound']
        elif self.model == "TextBlob":
            return TextBlob(text).sentiment.polarity

    def label_sentiment(self, score):
        #Labels sentiment based on the score and threshold
        if score >= self.threshold:
            return "Positive"
        elif score <= -self.threshold:
            return "Negative"
        else:
            return "Neutral"
        
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
        return f'SentimentModel:{self.model}'


if __name__ == "__main__":
    input_file = '../../data/train/processed/processed_data.csv'
    output_file = '../../data/train/processed/TextBlob_processed_data.csv'

    # Initialize with desired model ('VADER' or 'TextBlob')
    analyzer = SentimentAnalysis(input_file, output_file, model = "TextBlob")
    analyzer.process_tweets()

    df = pd.read_csv(output_file)

    biden_tweets = df[df['mentions_biden'] == 1]
    trump_tweets = df[df['mentions_trump'] == 1]

    if analyzer.model != 'ABSA':
        biden_sentiment_avg = biden_tweets['biden_sentiment_score'].mean()
        print("Biden - Average Sentiment Score:", biden_sentiment_avg)

    biden_sentiment_counts = biden_tweets['biden_sentiment_label'].value_counts()

    print("Biden - Sentiment Counts:\n", biden_sentiment_counts)
    print("===================")

    if analyzer.model != 'ABSA':
        trump_sentiment_avg = trump_tweets['trump_sentiment_score'].mean()
        print("Trump - Average Sentiment Score:", trump_sentiment_avg)

    trump_sentiment_counts = trump_tweets['trump_sentiment_label'].value_counts()

    print("Trump - Sentiment Counts:\n", trump_sentiment_counts)



