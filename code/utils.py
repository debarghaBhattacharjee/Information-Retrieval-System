# Add your import statements here
import pickle
import os
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

def save_data(data, file_name, directory):
    """
    Method to save data as a pickled string.
    """
    # Create directory if it does not exist.
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Save data as pickled object.
    pickle_out = open(
        f"{directory}/{file_name}", 
        "wb"
    )
    pickle.dump(data, pickle_out)
    pickle_out.close()
    
    msg = f"Model successfully saved: {directory}/{file_name}"
    print(msg)


def load_data(data_path):
    """
    Method to load data saved a s pickled string.
    """
    # Create directory if it does not exist.
    if not os.path.exists(data_path):
        print("Data doesn't exist.")
        return None
    
    # Retrieve the file.
    pickle_in = open(data_path,"rb")
    retrieved_data = pickle.load(pickle_in)
    pickle_in.close()
    return retrieved_data


def remove_stopwords(text):
    """
    Stopwords removal.

    Parameters
    ----------
    arg1 : list
    A list of lemmatized tokens.

    Returns
    -------
    list
    A new list with stopwords removed. 
    """

    stopword_removed_text = None

    #Fill in code here

    stop_words = set(stopwords.words('english'))
    cleaned_text = [
            tokens
            for tokens in text
            if tokens not in stop_words
    ]
    stopword_removed_text = cleaned_text

    return stopword_removed_text


def reduce_inflection(text):
    """
    Stemming/Lemmatization

    Parameters
    ----------
    arg1 : list
    A list of tokens (which are actually part of the same document).

    Returns
    -------
    list
    A list of stemmed/lemmatized tokens.
    """

    reduced_text = None
    #Fill in code here

    lemmatizer = WordNetLemmatizer()
    stemmed_tokens = [lemmatizer.lemmatize(tokens) for tokens in text]
    reduced_text = stemmed_tokens

    return reduced_text


def tokenize(text):
    """
    Tokenization using a Naive Approach

    Parameters
    ----------
    arg1 : str
    A string representing a document. 

    Returns
    -------
    list
    A list of tokens.
    """

    tokenized_text = None

    #Fill in code here
    tokenized_doc = [] 
    tokenized_doc = simple_preprocess(text)
    tokenized_text = tokenized_doc

    return tokenized_text


def text_cleanup(corpus, saving=False, file_name=None):
    """
    Perform text cleanup, which involves the following sub-tasks-
    1. Tokenizing the queries.
    2. Lemmatize the tokens.
    3. Removing the stopwords.
    """
    
    # Tokenize queries and saved list of tokenized queries.
    tokenized_corpus = []
    for doc in corpus:
        tokenized_doc = tokenize(doc)
        tokenized_corpus.append(tokenized_doc)
        
    # Stem/Lemmatize queries and save list of lemmatized queries.
    reduced_corpus = []
    for doc in tokenized_corpus:
        reduced_doc = reduce_inflection(doc)
        reduced_corpus.append(reduced_doc)
        
    # Remove stopwords from queries and save stopword removed queries.
    stopword_removed_corpus = []
    for doc in reduced_corpus:
        stopword_removed_doc = remove_stopwords(doc)
        stopword_removed_corpus.append(stopword_removed_doc)
        
    cleaned_corpus = stopword_removed_corpus
    
    # Save the cleaned up text.
    if saving==True:
        dir_path = "../output"
        save_data(cleaned_corpus, file_name, dir_path)
     
    return cleaned_corpus
    
    
    