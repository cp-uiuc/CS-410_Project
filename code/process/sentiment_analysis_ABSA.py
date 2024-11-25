import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm import tqdm
from setfit import AbsaModel
import nltk
import ssl
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

tqdm.pandas()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('vader_lexicon')

class SentimentAnalysis:
    # Extended to include ABSA columns
    COLS_KEEP = [
        'timestamp', 'country', 'candidate', 'is_en', 'sentiment_score', 
        'sentiment_label', 'likes', 'user_join_date', 'user_followers_count',
        'trump_label', 'trump_avg_polarity', 'biden_label', 'biden_avg_polarity', 'aspects'
    ]
    
    # Precompile regex patterns for ABSA
    TRUMP_PATTERN = re.compile(r'trump', re.IGNORECASE)
    BIDEN_PATTERN = re.compile(r'biden', re.IGNORECASE)
    
    def __init__(self, input_file_path, output_file_path, model="ALL", threshold=0.05, batch_size=32, num_workers=4):
        print("Initializing Sentiment Analysis...")
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.model = model
        self.threshold = threshold
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        print(f"Loading input data from: {input_file_path}")
        self.df = pd.read_csv(self.input_file_path)
        print(f"Data loaded successfully. Total rows: {len(self.df)}")

        if self.model in ["ALL", "VADER"]:
            print("Initializing VADER sentiment analyzer...")
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        if self.model in ["ALL", "ABSA"]:
            print("Loading ABSA model...")
            self.absa_model = AbsaModel.from_pretrained(
                "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect",
                "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity",
            )
            print("ABSA model loaded successfully.")

        self.polarity_scores = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        print("Initialization complete.")

    def process_tweets(self):
        print("Filtering English tweets...")
        self.df = self.df[self.df['is_en'] == True].copy()
        print(f"Filtered dataset. Total English tweets: {len(self.df)}")

        if self.model in ["ALL", "VADER", "TextBlob"]:
            print("Starting traditional sentiment analysis...")
            self.df['sentiment_score'] = self.df['textdata'].progress_apply(self.calculate_sentiment_score)
            self.df['sentiment_label'] = self.df['sentiment_score'].progress_apply(self.label_sentiment)
            print("Traditional sentiment analysis completed.")

        if self.model in ["ALL", "ABSA"]:
            print("Starting aspect-based sentiment analysis (ABSA)...")
            self._process_absa()
            print("Aspect-based sentiment analysis (ABSA) completed.")

        print("Saving results to output file...")
        self.df = self.df[self.COLS_KEEP]
        self.df.to_csv(self.output_file_path, index=False)
        print(f"Results saved to: {self.output_file_path}")
        self._print_analysis_results()

    def calculate_sentiment_score(self, text):
        if self.model == "VADER":
            return self.vader_analyzer.polarity_scores(text)['compound']
        elif self.model == "TextBlob":
            return TextBlob(text).sentiment.polarity
        elif self.model == "ALL":
            # Combine VADER and TextBlob scores
            vader_score = self.vader_analyzer.polarity_scores(text)['compound']
            textblob_score = TextBlob(text).sentiment.polarity
            return np.mean([vader_score, textblob_score])

    def label_sentiment(self, score):
        if score >= self.threshold:
            return "Positive"
        elif score <= -self.threshold:
            return "Negative"
        else:
            return "Neutral"

    def _process_absa(self):
        """Process ABSA in batches with an overall progress bar."""
        print("Splitting texts into batches for ABSA processing...")
        texts = self.df['textdata'].tolist()
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        print(f"Total batches created: {len(batches)}")

        all_results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            print(f"Processing batches with {self.num_workers} workers...")
            
            # Overall progress bar for all batches
            with tqdm(total=len(batches), desc="Processing all batches", unit="batch") as overall_progress:
                futures = {executor.submit(self._process_absa_batch, batch): i for i, batch in enumerate(batches)}

                for future in as_completed(futures):
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    
                    # Update the overall progress bar
                    overall_progress.update(1)

        print("Combining batch results into the DataFrame...")
        results_df = pd.DataFrame(
            all_results,
            columns=['trump_label', 'trump_avg_polarity', 'biden_label', 'biden_avg_polarity', 'aspects']
        )
        
        for col in results_df.columns:
            self.df[col] = results_df[col]

    def _process_absa_batch(self, texts):
        """Process a batch of texts for ABSA"""
        print(f"Processing a batch of size: {len(texts)}")
        results = []
        sentiments_batch = self.absa_model.predict(texts)
        
        for text, sentiments in zip(texts, sentiments_batch):
            trump_polarities = []
            biden_polarities = []
            aspects = []
            
            for entry in sentiments:
                if entry:
                    span_text = entry['span'].lower()
                    polarity_label = entry['polarity']
                    
                    if self.TRUMP_PATTERN.search(span_text):
                        if polarity_label in self.polarity_scores:
                            trump_polarities.append(self.polarity_scores[polarity_label])
                        aspects.append(entry['span'])
                    
                    if self.BIDEN_PATTERN.search(span_text):
                        if polarity_label in self.polarity_scores:
                            biden_polarities.append(self.polarity_scores[polarity_label])
                        aspects.append(entry['span'])
            
            trump_avg = np.mean(trump_polarities) if trump_polarities else None
            biden_avg = np.mean(biden_polarities) if biden_polarities else None
            
            results.append((
                self._calculate_absa_label(trump_avg),
                trump_avg,
                self._calculate_absa_label(biden_avg),
                biden_avg,
                aspects
            ))
        
        return results


    @staticmethod
    def _calculate_absa_label(avg_polarity):
        """Calculate ABSA sentiment label based on polarity"""
        if avg_polarity is None:
            return None
        elif avg_polarity > 0.1:
            return "positive"
        elif avg_polarity < -0.1:
            return "negative"
        else:
            return "neutral"

    def _print_analysis_results(self):
        """Print analysis results for both candidates"""
        df = pd.read_csv(self.output_file_path)
        
        # Traditional sentiment analysis results
        if 'sentiment_score' in df.columns:
            for candidate in ['biden', 'trump']:
                candidate_tweets = df[df['candidate'] == candidate]
                print(f"\n{candidate.upper()} - Traditional Sentiment Analysis:")
                print(f"Average Sentiment Score: {candidate_tweets['sentiment_score'].mean():.3f}")
                print("Sentiment Counts:\n", candidate_tweets['sentiment_label'].value_counts())
        
        # ABSA results
        if 'trump_label' in df.columns:
            print("\nABSA Results:")
            print("Trump Sentiment Distribution:", df['trump_label'].value_counts())
            print("Biden Sentiment Distribution:", df['biden_label'].value_counts())
            print("Trump Average Polarity:", df['trump_avg_polarity'].mean())
            print("Biden Average Polarity:", df['biden_avg_polarity'].mean())

if __name__ == "__main__":
    input_file = '../../data/train/processed/processed_data.csv'
    output_file = '../../data/train/processed/ABSA_processed_data.csv'
    
    # Initialize with "ALL" to use both traditional and ABSA analysis
    analyzer = SentimentAnalysis(
        input_file_path=input_file,
        output_file_path=output_file,
        model="ALL",
        batch_size=32,
        num_workers=4
    )
    analyzer.process_tweets()