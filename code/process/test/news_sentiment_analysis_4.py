import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from tqdm import tqdm
import json
from pandas import json_normalize
import spacy


tqdm.pandas()

# Download VADER lexicon
nltk.download('vader_lexicon')

class NewsSentimentAnalysis:

    COLS_KEEP = ['textdata', 'timestamp', 'mentions_trump', 'trump_sentiment_score', 'trump_sentiment_label', 'mentions_harris', 'harris_sentiment_score', 'harris_sentiment_label']
    
    def __init__(self, input_file_path, output_file_path, threshold=0.05):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.threshold = threshold
        self.sid = SentimentIntensityAnalyzer()
        self.nlp = spacy.load('en_core_web_sm')

        f = open(input_file_path)
        data =json.load(f)
        for a in data:
            self.df=pd.DataFrame(a,columns=['textdata', 'timestamp', 'mentions_trump', 'mentions_harris'])
    
    def process_news_textdata(self):

        print("Calculating polarity scores and labeling sentiments for each article title + description...")
        print(self.df)
        
        # Apply VADER sentiment analysis to the entire textdata (title + description)
        self.df['sentiment_score'] = self.df['textdata'].progress_apply(
            lambda text: self.sid.polarity_scores(text)['compound']
        )
        self.df['sentiment_label'] = self.df['sentiment_score'].progress_apply(self.label_sentiment)

        # Perform aspect-based sentiment analysis for Trump and Harris
        self.df['trump_sentiment_score'] = self.df['textdata'].progress_apply(
            lambda text: self.get_aspect_sentiment(text, 'trump', self.df['timestamp'])
        )
        self.df['trump_sentiment_label'] = self.df['trump_sentiment_score'].progress_apply(self.label_sentiment)

        self.df['harris_sentiment_score'] = self.df['textdata'].progress_apply(
            lambda text: self.get_aspect_sentiment(text, 'harris', self.df['timestamp'])
        )
        self.df['harris_sentiment_label'] = self.df['harris_sentiment_score'].progress_apply(self.label_sentiment)

        # Filter the relevant columns
        self.df = self.df[self.COLS_KEEP]
        self.df.to_csv(self.output_file_path, index=False)

    def get_aspect_sentiment(self, text, aspect, timestamp):
        """Extracts sentiment for a specific aspect using dependency parsing."""
        # Process text with spaCy for dependency parsing
        doc = self.nlp(text)

        # Extract relevant phrases related to the aspect
        relevant_phrases = []
        for token in doc:
            if aspect.lower() in token.text.lower():  # Case insensitive match for aspect
                # Add token's surrounding context (dependencies)
                relevant_phrases.append(self.extract_dependency_context(token))
        
        # Join all relevant phrases into one text and analyze sentiment
        relevant_text = ' '.join(relevant_phrases)
        print(aspect + " " + str(timestamp) + " " + relevant_text)
        return self.sid.polarity_scores(relevant_text)['compound']
    
    def extract_dependency_context(self, token):
        """Extracts the context around the entity from the dependency tree."""
        context = []
        
        # Go leftwards to collect words (subject or modifier)
        for left in token.lefts:
            context.append(left.text)
        
        # Add the token itself
        context.append(token.text)
        
        # Go rightwards to collect words (object or modifier)
        for right in token.rights:
            context.append(right.text)
        
        # Return the combined context
        return ' '.join(context)

    def label_sentiment(self, score):
        """Labels sentiment based on the score and threshold."""
        if score >= self.threshold:
            return "Positive"
        elif score <= -self.threshold:
            return "Negative"
        else:
            return "Neutral"


# Example usage
input_file = '../../../data/test/processed/processed_election_news.json'
output_file = '../../../data/test/processed/VADER_processed_news_data.csv'
analyzer = NewsSentimentAnalysis(input_file, output_file)
analyzer.process_news_textdata()