To design a complex email spam detection system in Python, we’ll follow several steps that involve data preprocessing, feature extraction, model training, and evaluation. We’ll use machine learning techniques with libraries like `scikit-learn` and `NLTK`, as well as NLP techniques to extract useful features from email content.

Here's a step-by-step guide:

### Step 1: Import Required Libraries
We’ll need libraries for text processing, data manipulation, and machine learning.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import re
```

Make sure you have the necessary packages installed (`nltk`, `scikit-learn`, `pandas`, etc.).

### Step 2: Load and Preprocess Data
We’ll use a dataset of emails labeled as spam or not spam. For demonstration, you can use a dataset like the [SpamAssassin](https://spamassassin.apache.org/publiccorpus/) dataset.

```python
# Load dataset
data = pd.read_csv("spam_data.csv")  # Ensure your dataset is in the same directory

# Display dataset information
print(data.head())
print(data['label'].value_counts())  # Check distribution of spam vs. non-spam

# Map labels to binary format
data['label'] = data['label'].map({'spam': 1, 'ham': 0})
```

### Step 3: Data Cleaning and Text Preprocessing
Clean the email text to remove unnecessary symbols, stopwords, and punctuations.

```python
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

data['text'] = data['text'].apply(clean_text)
```

### Step 4: Feature Extraction with TF-IDF
Convert the cleaned email text to numeric features using the TF-IDF vectorizer.

```python
tfidf = TfidfVectorizer(max_features=3000)  # Limit features for performance
X = tfidf.fit_transform(data['text']).toarray()
y = data['label'].values
```

### Step 5: Split the Data for Training and Testing
We’ll split our data into a training set and a test set.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Step 6: Train Machine Learning Models
We’ll use both Naive Bayes and Random Forest classifiers to identify which performs better for spam detection.

```python
# Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

### Step 7: Make Predictions and Evaluate the Models
Evaluate both models using accuracy, precision, recall, and F1 score.

```python
# Naive Bayes Evaluation
nb_predictions = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
print("Naive Bayes Report:\n", classification_report(y_test, nb_predictions))

# Random Forest Evaluation
rf_predictions = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("Random Forest Report:\n", classification_report(y_test, rf_predictions))
```

### Step 8: Optimize the Model (Optional)
Experiment with feature engineering and parameter tuning for better performance.

1. **Hyperparameter tuning:** Use `GridSearchCV` to tune hyperparameters of models.
2. **Feature selection:** Consider adding bigrams, trigrams, or more sophisticated NLP features.
3. **Ensemble methods:** Experiment with an ensemble of models to improve prediction accuracy.

### Step 9: Implementing Real-time Spam Detection (Optional)
After training, you can use the trained model in real-time by creating a function to predict whether a new email is spam or not:

```python
def predict_spam(email_text):
    cleaned_text = clean_text(email_text)
    text_features = tfidf.transform([cleaned_text]).toarray()
    prediction = rf_model.predict(text_features)
    return "Spam" if prediction == 1 else "Not Spam"

# Test on a new email
new_email = "Congratulations! You've won a free cruise. Call now to claim your prize."
print(predict_spam(new_email))
```

### Summary
This setup includes the following:
1. **Data Preprocessing:** Cleaning and preparing text.
2. **Feature Extraction:** Converting text into numerical features with TF-IDF.
3. **Model Training:** Training models with Naive Bayes and Random Forest.
4. **Evaluation:** Assessing model accuracy and tuning where necessary.
5. **Real-time Detection:** Function to predict new emails as spam or not.

This system can serve as a robust foundation for spam detection. 
