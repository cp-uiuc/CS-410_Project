import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm import tqdm

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

class SentimentAnalysis:

    COLS_KEEP = ['timestamp', 'country', 'candidate', 'is_en', 'sentiment_score', 'sentiment_label', 'likes', 'user_join_date', 'user_followers_count']

    def __init__(self, input_file_path, output_file_path, model = "VADER", threshold=0.05):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.model = model
        self.threshold = threshold
        self.df = pd.read_csv(self.input_file_path)

        if self.model == "VADER":
            self.analyzer = SentimentIntensityAnalyzer()
        elif self.model == "TextBlob":
            self.analyzer = None
        else:
            raise ValueError("Invalid model choice. Choose 'VADER' or 'TextBlob'.")

    def process_tweets(self):
        # Filter English tweets
        self.df = self.df[self.df['is_en'] == True].copy()

        print("Calculating polarity scores and labeling sentiments for each tweet...")
        self.df['sentiment_score'] = self.df['textdata'].progress_apply(self.calculate_sentiment_score)
        self.df['sentiment_label'] = self.df['sentiment_score'].progress_apply(self.label_sentiment)
        self.df = self.df[self.COLS_KEEP]
        self.df.to_csv(self.output_file_path, index=False)

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
    
    def __repr__(self):
        return f'SentimentModel:{self.model}'


if __name__ == "__main__":
    input_file = '../../data/train/processed/processed_data.csv'
    output_file = '../../data/train/processed/VADER_processed_data.csv'

    # Initialize with desired model ('VADER' or 'TextBlob')
    analyzer = SentimentAnalysis(input_file, output_file, model = "VADER")
    analyzer.process_tweets()

    df = pd.read_csv(output_file)

    biden_tweets = df[df['candidate'] == 'other']
    trump_tweets = df[df['candidate'] == 'trump']

    biden_sentiment_avg = biden_tweets['sentiment_score'].mean()
    biden_sentiment_counts = biden_tweets['sentiment_label'].value_counts()

    print("Biden - Average Sentiment Score:", biden_sentiment_avg)
    print("Biden - Sentiment Counts:\n", biden_sentiment_counts)
    print("===================")

    trump_sentiment_avg = trump_tweets['sentiment_score'].mean()
    trump_sentiment_counts = trump_tweets['sentiment_label'].value_counts()

    print("Trump - Average Sentiment Score:", trump_sentiment_avg)
    print("Trump - Sentiment Counts:\n", trump_sentiment_counts)



