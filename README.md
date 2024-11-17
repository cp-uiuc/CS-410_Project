# CS410 Course Project

## How to Run the Project

## For Training Model

1. **Clone the Repository**: Pull this project to your local machine using Git.

2. **Download Dataset**: Retrieve the *US Election 2020 Tweets* dataset from [Kaggle](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets/data).

3. **Organize Files**: Place `hashtag_donaldtrump.csv` and `hashtag_joebiden.csv` in the `../../data/train/raw` directory.

4. **Set Up Environment**:
   - **Mac Users**: In the directory where `environment.yml` is located, create the environment using:
     ```bash
     conda env create -f environment.yml
     ```
   - **Windows Users**: Manually create a Conda environment and install the required packages:
     ```bash
     conda create -n cs410_project
     conda activate cs410_project
     pip install pandas spacy nltk statsmodels
     ```

5. **Initialize Data Processor**: Run `process_data.py` to initialize the `DataProcessor` class.

6. **Data Cleaning and Labeling**: Execute `script.py` to clean and label the data using the `DataProcessor` class. 
   - *Note*: This step may take approximately 5 hours. You should see a `processed_data.csv` file after successful completion of this step.

7. **Data Cleaning and Labeling after adding Sentiment Analysis**: Run `TrainPipeline.py` to configure sentiment analysis, process sentiment data, and generate predictions using the defined pipeline.

*Note* First cd into process/code before running any scripts


## For Testing Model