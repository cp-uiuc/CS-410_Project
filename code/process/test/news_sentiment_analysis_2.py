from typing import Counter
import pandas as pd
import spacy
from setfit import AbsaModel
from tqdm import tqdm
import json
import re

tqdm.pandas()

nlp = spacy.load('en_core_web_lg')


class NewsSentimentAnalysis:

    COLS_KEEP = ['textdata', "trump_label", "trump_avg_polarity", "harris_label", "harris_avg_polarity", "aspects"]
    
    def __init__(self, input_file_path, output_file_path, threshold=0.05):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.threshold = threshold

        self.sentiment_model = AbsaModel.from_pretrained(
            "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect",
            "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity",
        )

         # Load the data
        f = open(input_file_path)
        data =json.load(f)
        for a in data:
            self.df=pd.DataFrame(a,columns=['textdata', 'timestamp'])

    def process_news_textdata(self):
        """Calculate sentiment labels for specific entities."""
        print("Performing sentiment analysis...")
        self.df[["trump_label", "trump_avg_polarity", "harris_label", "harris_avg_polarity", "aspects"]] = self.df['textdata'].progress_apply(
            self.analyze_sentiment
        )
        
        # Keep only relevant columns
        self.df = self.df[self.COLS_KEEP]

        # Save the results to a CSV file
        self.df.to_csv(self.output_file_path, index=False)
        print(f"Sentiment analysis results saved to {self.output_file_path}")

    def analyze_sentiment(self, text):
        """
        Extract sentiment labels based on average polarity for Trump and Harris from the text.

        Args:
            text (str): The text to analyze.

        Returns:
            pd.Series: A Series containing Trump label, Trump average polarity, Harris label,
                    Harris average polarity, and aspects mentioned.
        """
        sentiments = self.sentiment_model.predict(text)  # Assume this gives a list of dictionaries

        # Initialize variables to track polarities and aspects for Trump and Harris
        trump_polarities = []  # List to store numeric polarities for Trump
        harris_polarities = []  # List to store numeric polarities for Harris
        aspects = []  # Store all spans mentioning Trump or Harris

        # Mapping of polarities to numeric scores
        polarity_scores = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

        # Loop through the sentiment entries
        for entry in sentiments:
            if entry:  # Ensure entry is not empty
                span_text = entry['span'].lower()
                polarity_label = entry['polarity']
                
                if 'trump' in span_text:
                    # Capture polarity for Trump
                    if polarity_label in polarity_scores:
                        trump_polarities.append(polarity_scores[polarity_label])
                    aspects.append(entry['span'])

                if 'harris' in span_text:
                    # Capture polarity for Harris
                    if polarity_label in polarity_scores:
                        harris_polarities.append(polarity_scores[polarity_label])
                    aspects.append(entry['span'])

        # Calculate average polarities
        trump_avg_polarity = (sum(trump_polarities) / len(trump_polarities)) if trump_polarities else None
        harris_avg_polarity = (sum(harris_polarities) / len(harris_polarities)) if harris_polarities else None

        # Function to determine label based on average polarity
        def calculate_label(avg_polarity):
            if avg_polarity is None:
                return None
            elif avg_polarity > 0.1:  # Threshold for positive
                return "positive"
            elif avg_polarity < -0.1:  # Threshold for negative
                return "negative"
            else:  # Threshold for neutral
                return "neutral"

        # Assign labels based on average polarity
        trump_label = calculate_label(trump_avg_polarity)
        harris_label = calculate_label(harris_avg_polarity)

        # Return the labels, average polarities, and aspects as a Pandas Series
        return pd.Series([trump_label, trump_avg_polarity, harris_label, harris_avg_polarity, aspects])

# Paths to input and output files
input_file = '../../../data/test/processed/processed_election_news.json'
output_file = '../../../data/test/processed/ABSA_processed_news_data.csv'

# Instantiate and process data
analyzer = NewsSentimentAnalysis(input_file, output_file)
analyzer.process_news_textdata()