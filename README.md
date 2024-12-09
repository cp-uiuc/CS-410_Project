# CS410 Course Project

## How to Run the Project

1. **Clone the Repository**: Pull this project to your local machine using Git.

2. **Set Up Environment**:
   - **Mac Users**: In the directory where `environment.yml` is located, create the environment using:
     ```bash
     conda env create -f environment.yml
     ```
     *Note*: Mac users **cannot** run the Training sentiment analysis for ABSA (Aspect-Based Sentiment Analysis) because it requires CUDA for GPU acceleration, which is not supported on macOS. However, the pre-processed sentiment analysis output file is available in UIUC Box and is automatically used by the script

   - **Windows Users**: Manually create a Conda environment and install the required packages:
     ```bash
     conda create -n cs410_project python=3.11
     conda activate cs410_project

     conda install pandas tqdm numpy=1.26
     conda install -c conda-forge spacy
     conda install -c conda-forge nltk
     conda install -c conda-forge textblob
     conda install -c conda-forge statsmodels
     conda install -c conda-forge scikit-learn

     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     pip3 install spacy-langdetect transformers[sentencepiece] tiktoken protobuf
     ```
     *Note*: Windows users **must** have the CUDA toolkit installed to run the Training sentiment analysis for ABSA, as the model requires GPU acceleration. However, the pre-processed sentiment analysis output file is available in UIUC Box and is automatically used by the script.

3. **For Train Tweet Data**:
   a. Navigate to the `code/process/train` folder:
      - Run `python process_data.py` to initialize the 'DataProcessor' class.
      - Run `python script.py` to clean and label the data using the 'DataProcessor' class.
        - *Note*: This step may take approximately 5 hours. You should see a `processed_data.csv` file after successful completion of this step.
   b. Navigate to the `code/pipeline` folder:
      - Run `python TrainPipeline.py` to configure sentiment analysis, process sentiment data, and generate predictions using the defined pipeline.

4. **For Test News Data**:
   a. Navigate to the `code/process/test` folder:
      - Run `python news_process_data.py` to initialize the 'NewsDataProcessor' class.
      - Run `python script.py` to clean and label the data using the 'NewsDataProcessor' class.
        - *Note*: You should see a `processed_data.csv` file after successful completion of this step.
   b. Navigate to the `code/pipeline` folder:
      - Run `python TestPipeline.py` to configure sentiment analysis, process sentiment data, and generate predictions using the defined pipeline.

**Flow:**
- `process_data.py` → `script.py` → `sentiment_analysis.py` or `news_sentiment_analysis.py` → `process_layer2_data.py` or `news_process_layer2_data.py` → `model.py`  
  *(The last three files were combined together in `TrainPipeline.py`/`TestPipeline.py`)*

**Note**: Some scripts (e.g., `process_data.py`) in the Training flow take hours to run. Instead of requiring users to run those scripts, pre-processed outputs are stored in the cloud (UIUC Box) to use as data sources.
