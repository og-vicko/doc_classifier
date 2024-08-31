import contractions
import re 
import string
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

def expand_contractions(text):
    """
    Expand literal word contractions (e.g I'm not happy --> I am not happy. It's terrible --> It is terrible)
    """
    expanded_contraction = contractions.fix(text, slang=False)
    return expanded_contraction


def clean_words(text):
    """
    Clean the text in the News Article
    """
    text = re.sub(r"[^a-zA-Z ]+", " ", text.strip())         #<--- remove non-alphabetic words
    text = re.sub(r'\w*\d\w*', '', text)                     #<--- Remove alphanumeric words
    text = re.sub(r"\s{2,}", " ", text)                      #<--- Replace multiple spaces with a single space
    text = re.sub(r"https?://\S+|www\.\S+", "", text)        #<--- Remove URLs (http/https and www)
    text = re.sub('<.*?>', '', text)                         #<--- Remove HTML tags
    
    
    text = ''.join([char for char in text if char not in string.punctuation])     #<--- Remove punctuation
    text = text.replace('\n', ' ')                                                #<--- Remove newline characters
    
    return text


def tokenize(text):
    """
    Tokenize the input text.
    Args:
        text (str): The input text to be tokenized.
    Returns:
        str: A space-separated string of tokens.
    Example:
        >>> text = "This is a sample sentence."
        >>> tokenize(text)
        ['This', 'is', 'a', 'sample', 'sentence', '.']
    """
    lowercase = text.lower()                           #<--- Convert text to lower case
    tokens = word_tokenize(lowercase)                  #<--- Perform tokenization
    return (" ".join(tokens)).strip()                  #<--- Join tokens with spaces and strip extra spaces


def remove_stopwords(text):
    """
    Remove stopwords from the input text.
    """
    nltk_stop_words = nltk.corpus.stopwords.words('english')                    #<--- Get the list of English stopwords
    word_list = [word for word in text.split() if word not in nltk_stop_words]  #<--- Remove stopwords from the text
    return " ".join(word_list)                                                  #<--- Join remaining words into a string


def lemmatize(text):
    """
    Lemmatize the words in the input text.
    """
    lemma = WordNetLemmatizer()                                        #<--- Initialize the WordNet lemmatizer
    lemmatized = [lemma.lemmatize(word) for word in text.split()]      #<--- Lemmatize each word in the text
    return " ".join(lemmatized)                                        #<--- Join the lemmatized words into a string


def preprocess_text(text):
    """
    Preprocesses the input text using a series of text cleaning and normalization steps.
    Args:
        text (str): The input text to be preprocessed.
    Returns:
        str: Preprocessed text.
    """
    # print(f"Type of input text: {type(text)}")

    text = expand_contractions(text)
    text = clean_words(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    
    return text

def preprocess_data(text_series):
    """
    Preprocesses the input text using a series of text cleaning and normalization steps.
    Args:
        text_series (pd.Series or str): The input text to be preprocessed.
    Returns:
        pd.Series or str: Preprocessed text.
    """
    if isinstance(text_series, pd.Series):
        return text_series.apply(lambda text: preprocess_text(text))
    elif isinstance(text_series, str):
        return preprocess_text(text_series)
    else:
        raise ValueError("Input must be a pandas Series or a string.")
