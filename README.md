# Sentiment Analysis of IMDB Movie Reviews

This project demonstrates sentiment analysis using Natural Language Processing (NLP) on a dataset of IMDB movie reviews. 

## Project Overview

The goal of this project is to build a model that can accurately predict the sentiment (positive or negative) of a movie review.  The project utilizes the following steps:

1. **Data Loading and Exploration:**  The IMDB movie review dataset is loaded and explored using Pandas.
2. **Data Preprocessing:**  Text data is cleaned and prepared for analysis by:
    * Tokenizing the text into individual words.
    * Removing stop words (common words like "the", "a", "is").
    * Converting words to lowercase.
3. **Train-Test Split:** The dataset is divided into training and testing sets to evaluate the model's performance.
4. **Feature Extraction:** The CountVectorizer is used to convert text data into numerical features (Bag-of-Words model).
5. **Model Building and Training:**  A Multinomial Naive Bayes classifier is trained on the training data.
6. **Prediction and Evaluation:** The trained model is used to predict the sentiment of the test data, and its performance is evaluated using metrics like accuracy, precision, recall, and F1-score.

## Dependencies

* Python 3.x
* Pandas
* NumPy
* NLTK
* Scikit-learn

## Installation

1. Install the required libraries:
2. Download NLTK resources:

## Usage

1. Place the IMDB movie review dataset (`IMDB_dataset-1.csv`) in the same directory as the code.
2. Run the Python code to train the model, make predictions, and evaluate the results.

## Results

The model achieves an accuracy of approximately 86% in predicting the sentiment of movie reviews. This indicates good performance in classifying reviews as positive or negative. Detailed evaluation metrics are provided in the code output.

## Further Improvements

* Explore more advanced text preprocessing techniques.
* Experiment with different machine learning models.
* Fine-tune model hyperparameters for better performance.
* Incorporate more features like sentiment lexicons or part-of-speech tags.
   
