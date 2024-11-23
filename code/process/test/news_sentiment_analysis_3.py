from joblib import Parallel, delayed
import pandas as pd
import nltk
from tqdm import tqdm
import json
from transformers import pipeline
tqdm.pandas()

# Initialize the zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

class NewsSentimentAnalysis:

    COLS_KEEP = ['textdata', 'timestamp','trump_label', 'biden_label']
    
    def __init__(self, input_file_path, output_file_path, labels=["positive", "neutral", "negative"]):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.labels = labels
        f = open(input_file_path)
        data =json.load(f)
        for a in data:
            self.df=pd.DataFrame(a,columns=['textdata', 'timestamp'])
    
    def process_news_textdata(self):
        print("Classifying sentiment for each news article...")
        # Use parallel processing with tqdm to show progress
        results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(self.get_sentiment)(text, person) 
            for text in tqdm(self.df['textdata'], desc="Processing articles", position=0) 
            for person in ['Donald Trump', 'Kamala Harris']
        )
        # Unpack results into separate columns
        self.df['donald_trump_label'], self.df['kamala_harris_label'] = zip(*[results[i:i+2] for i in range(0, len(results), 2)])

        # Save results to CSV
        self.df.to_csv(self.output_file_path, index=False)
    
    def get_sentiment(self, text, person):
        """Classifies sentiment towards a specific person (e.g., Donald Trump or Kamala Harris)."""
        result = classifier(f"What is the sentiment toward {person} in: {text}", candidate_labels=self.labels)
        return result['labels'][result['scores'].index(max(result['scores']))]

if __name__ == '__main__':
    # File paths
    input_file = '../../../data/test/processed/processed_election_news.json'
    output_file = '../../../data/test/processed/ZSC_processed_news_data.csv'

    # Create and process the news sentiment analysis
    analyzer = NewsSentimentAnalysis(input_file, output_file)
    analyzer.process_news_textdata()
