# IMDb Movie Review Classification

This project implements a machine learning model to classify movie reviews as positive or negative based on their content. The model is trained using a dataset of movie reviews and their corresponding labels.

## Requirements

- pandas
- scikit-learn
- nltk
- joblib

You can install the required packages using the following command:

```sh
pip install -r louis_requirements.txt
```

## Implementation Details
### 1. Importing Libraries
The following libraries are used in this project:

pandas: For data manipulation and loading the CSV file.
scikit-learn: For machine learning tasks including vectorization, model training, and evaluation.
NLTK: For natural language processing tasks such as tokenization, lemmatization, and stop word removal.
Regular Expressions (re): For text preprocessing to remove unwanted characters.
joblib: For saving and loading the trained model.
os: For checking if the model file exists.

### 2. Downloading NLTK Data Files
The necessary NLTK data files are downloaded for tokenization, lemmatization, and stop word removal. These downloads only need to be run once.

### 3. Loading the CSV Data
The dataset is loaded into a pandas DataFrame for easy manipulation and analysis.

### 4. Preprocessing the Data
The features (user_review) and labels (status) are extracted from the dataset. The labels are encoded into numerical values using LabelEncoder for compatibility with the machine learning model.