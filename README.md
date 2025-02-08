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

* pandas: For data manipulation and loading the CSV file.
* scikit-learn: For machine learning tasks including vectorization, model training, and evaluation.
* NLTK: For natural language processing tasks such as tokenization, lemmatization, and stop word removal.
* Regular Expressions (re): For text preprocessing to remove unwanted characters.
* joblib: For saving and loading the trained model.
* os: For checking if the model file exists.

### 2. Downloading NLTK Data Files
The necessary NLTK data files are downloaded for tokenization, lemmatization, and stop word removal. These downloads only need to be run once.

### 3. Loading the CSV Data
The dataset is loaded into a pandas DataFrame for easy manipulation and analysis.

### 4. Preprocessing the Data
The features ```(user_review)``` and labels ```(status)``` are extracted from the dataset. The labels are encoded into numerical values using LabelEncoder for compatibility with the machine learning model.

### 5. Text Preprocessing Function
A function is defined to preprocess the text data:

* Lemmatization: Reduces words to their base form, improving consistency.
* Stop Words Removal: Removes common words that do not contribute much to the meaning.
* Regular Expressions: Cleans the text by removing HTML tags and special characters.
* Tokenization: Splits the text into individual words.

### 6. Applying Text Preprocessing
The preprocessing function is applied to all reviews to clean and standardize the text data.

### 7. Splitting the Data
The data is split into training and testing sets to evaluate the model's performance on unseen data. 80% of the data is used for training and 20% for testing.

### 8. Creating and Training the Model
A pipeline is created that combines TfidfVectorizer and SVC for streamlined processing:

1. TF-IDF Vectorization: Converts text data into numerical features, capturing the importance of words.
2. SVM Classifier: A powerful classifier that works well with high-dimensional data.

<img src="https://ch.mathworks.com/discovery/support-vector-machine/_jcr_content/mainParsys/image.adapt.full.medium.jpg/1718266259806.jpg" alt="drawing" width="200"/>

3. Grid Search: Hyperparameter tuning to find the best combination of parameters for the model.
4. Model Persistence: The trained model is saved to a file using joblib to avoid retraining in future runs.

### 9. Evaluating the Model
The model's accuracy is evaluated by predicting the labels for the test set and comparing them to the true labels.

### 10. Predicting the Status of a New Review
A function is defined to preprocess and predict the status of a new review. The while True loop continuously prompts the user for a review, providing an option to exit the loop by typing 'exit'.

## Usage
To run the script, use the following command:

```sh
python louis_classification.py
```

The script will load the dataset, preprocess the data, train the model (if not already saved), and prompt the user to enter movie reviews for classification.
