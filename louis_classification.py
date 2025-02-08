import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data files (only need to run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load the CSV data
print("Loading CSV data...")
data = pd.read_csv('reviews.csv')
print(f"Loaded {len(data)} records.")

# Preprocess the data
# Assuming 'user_review' is the text of the review and 'status' is the label
X = data['user_review']
y = data['status']

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Define a function for text preprocessing
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Apply text preprocessing
print("Preprocessing text data...")
X = X.apply(preprocess_text)

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {len(X_train)} records")
print(f"Testing set: {len(X_test)} records")

# Create a pipeline that combines a TfidfVectorizer with an SVM classifier
print("Creating and training the model...")
model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))

# Train the model
model.fit(X_train, y_train)
print("Model training completed.")

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

# Example usage
new_review = "C'est un film profondément émouvant que je recommande à tout le monde. Le travail de la caméra est excellent et les performances sont déterminantes pour ma carrière. Une réussite incroyable."
print(f'The review status is: {predict_review_status(new_review)}')