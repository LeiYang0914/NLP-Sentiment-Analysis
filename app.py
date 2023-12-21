import streamlit as st
import joblib
from googletrans import Translator
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from googletrans import Translator
import re
import unicodedata
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

def translate_text(text):
    translator = Translator()
    try:
        translated = translator.translate(text, src='auto', dest='en')
        return translated.text
    except Exception as e:
        print(f"Error during translation: {e}")
        return text    

abbreviation_dict = {
    'utk': 'untuk',
    'mmg': 'memang',
    'jdi': 'jadi',
    'smpai': 'sampai',
    'mcm': 'macam',
    'sbb': 'sebab',
    'tq': 'thanks',
    'yg': 'yang',
    'sgt': 'sangat'
}

# Initialize the translator and the lemmatizer
translator = Translator()
lemmatizer = WordNetLemmatizer()

def replace_abbreviations(text, abbreviation_dict):
    words = text.split()
    replaced_text = ' '.join([abbreviation_dict.get(word, word) for word in words])
    return replaced_text

def replace_supersub(text):
    super_regex = re.compile(r'\u00B2|\u00B3|\u00B9|\u2070-\u2079|\u2080-\u2089|\u2460-\u2469|\u2474-\u247D|\u2488-\u2491|\u24F5-\u24FE|\u2776-\u277F|\u2780-\u2793|\u3192-\u319F|\u3220-\u3229|\u3280-\u3289')
    sub_regex = re.compile(r'[\u2080-\u2089]')
    text = super_regex.sub(lambda m: str(unicodedata.digit(m.group())), text)
    text = sub_regex.sub(lambda x: str(unicodedata.digit(x.group())), text)
    return text

# Function to map NLTK's POS tags to WordNet POS tags
def nltk_pos_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
    
# Function to lemmatize text using POS tags
def lemmatize_text(text, lemmatizer):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

# Define the non-English character pattern
non_english_char_pattern = re.compile(
    r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df'
    r'\U0002a700-\U0002ebef\U00030000-\U000323af\ufa0e\ufa0f'
    r'\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27\ufa28\ufa29'
    r'\u3006\u3007][\ufe00-\ufe0f\U000e0100-\U000e01ef]?'
)

# Function to check for non-English characters
def contains_non_english_characters(text):
    if not isinstance(text, str):
        return False
    return re.search(non_english_char_pattern, text) is not None

def preprocess_english_text(text, abbreviation_dict):
   # Check if the text is a string
    if not isinstance(text, str):
        return ""
    
    # Replace abbreviations
    text = replace_abbreviations(text, abbreviation_dict)

    # Replace superscript and subscript characters
    text = replace_supersub(text)
    
    # Lowercasing
    text = text.lower()

    # Check for non-English characters  
    if contains_non_english_characters(text):
        # skip texts with non-English characters
        return ""

    # Removing numbers and punctuations
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    # Lemmatization using POS tags
    text = lemmatize_text(text, lemmatizer)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def classify_review(user_review, pipeline, svm_model):
    translate_user_review = translate_text(user_review)
    processed_review = preprocess_english_text(translate_user_review, abbreviation_dict)
    transformed_review = pipeline.transform([processed_review])
    predicted_sentiment = svm_model.predict(transformed_review)
    return predicted_sentiment[0]

def load_models():
    pipeline = joblib.load('text_processing_pipeline.pkl')
    svm_model = joblib.load('svm_model.pkl')
    return pipeline, svm_model

def main():
    # Page configuration
    st.set_page_config(page_title="Review Sentiment Analysis", layout="wide")

    st.title("Review Sentiment Analysis")

    # Load models
    pipeline, svm_model = load_models()

    # Text input
    user_review = st.text_area("Enter Your Review", height=150)

    if st.button("Analyze Sentiment"):
        if user_review:
            translated_review = translate_text(user_review)
            st.write("Translated Review:", translated_review)
            sentiment = classify_review(user_review, pipeline, svm_model)
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error("Please enter a review to analyze.")

if __name__ == "__main__":
    main()
