
# Spam Email Classification Using NLP and Neural Networks

This project demonstrates the full pipeline of building a spam email classifier using natural language processing (NLP) techniques and a neural network model.

##  Dataset

The dataset is loaded from a CSV file named `emails.csv` and contains two columns:

- `Message`: the email text content
- `Spam`: 1 if the message is spam, 0 otherwise

##  Steps

### 1. Data Loading and Preprocessing

- The data is read using `pandas`
- Tokenization, stopword removal, and stemming are performed using `nltk`
- Custom preprocessing function cleans the text (removing punctuation, URLs, digits)

### 2. Text Visualization

- Word cloud is generated for spam messages using `wordcloud` and `matplotlib`

### 3. Feature Extraction

- Bag of Words (CountVectorizer) and TF-IDF (TfidfVectorizer) representations are used
- A maximum of 100 features is extracted

### 4. Model Training

- A simple feedforward neural network is trained using Keras with:
  - Input layer with 64 units
  - Hidden layer with 32 units
  - Output layer with sigmoid activation for binary classification

- The model is trained and evaluated using both CountVectorizer and TfidfVectorizer features

### 5. Prediction

- A custom message is preprocessed and classified by the trained model

##  Requirements

- pandas
- nltk
- matplotlib
- wordcloud
- scikit-learn
- tensorflow / keras

Make sure to install the required packages before running the notebook or script.

##  Output

- Model accuracy is printed after training on both BoW and TF-IDF features.
- Final prediction is shown for a sample spam message.

##  Example Result

```
Accuracy: 97.25
Prediction for custom message: [1.]
```

---
*This project shows how deep learning and NLP can be combined for text classification tasks like spam detection.*
