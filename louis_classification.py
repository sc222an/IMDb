import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import re
import joblib
import os

# Download NLTK data files (only need to run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# Load the training CSV data
print("Loading training CSV data...")
train_data = pd.read_csv('reviews.csv')
print(f"Loaded {len(train_data)} training records.")

# Load the testing CSV data with specified column names
print("Loading testing CSV data...")
column_names = ['const', 'title_id', 'title_name', 'user_review', 'status', 'sub_date']
test_data = pd.read_csv('test-reviews.csv', names=column_names, header=None)
print(f"Loaded {len(test_data)} testing records.")

# Preprocess the training data
# Assuming 'user_review' is the text of the review and 'status' is the label
X_train = train_data['user_review']
y_train = train_data['status']

# Preprocess the testing data
X_test = test_data['user_review']
y_test = test_data['status']

# Encode the labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Define a function for text preprocessing
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Apply text preprocessing
print("Preprocessing training text data...")
X_train = X_train.apply(preprocess_text)

print("Preprocessing testing text data...")
X_test = X_test.apply(preprocess_text)

# Create a pipeline that combines a TfidfVectorizer with an SVM classifier
print("Creating the model pipeline...")
pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), SVC(kernel='linear', class_weight='balanced'))

# Hyperparameter tuning using Grid Search
param_grid = {
    'tfidfvectorizer__max_df': [0.75, 1.0],
    'tfidfvectorizer__max_features': [None, 5000],
    'svc__C': [0.1, 1]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)

# Check if the model already exists
if os.path.exists('sentiment_model.pkl'):
    print("Loading the saved model...")
    model = joblib.load('sentiment_model.pkl')
else:
    print("Training the model...")
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    joblib.dump(model, 'sentiment_model.pkl')
    print("Model training completed and saved.")

# Predict the status of the test set
print("Predicting the status of the test set...")
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Function to predict the status of a new review
def predict_review_status(review):
    print(f"Predicting status for review: {review}")
    preprocessed_review = preprocess_text(review)
    prediction = model.predict([preprocessed_review])[0]
    return label_encoder.inverse_transform([prediction])[0]

# Prompt the user for a review in a loop
while True:
    new_review = input("Please enter a movie review (or type 'exit' to quit): ")
    if new_review.lower() == 'exit':
        break
    print(f'The review status is: {predict_review_status(new_review)}')