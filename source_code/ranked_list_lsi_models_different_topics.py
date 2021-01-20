#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *

import json
import pickle
import os
import numpy as np

from gensim import corpora
from gensim import models
from gensim import similarities


# In[2]:


"""
Try out LSI for different models of k.
"""
# Load the preprocessed docs.
docs = load_data("../output/cleaned_docs.pickle")

# Load the preprocessed queries.
queries = load_data("../output/cleaned_queries.pickle")

# Load the dictionary .
in_file = "dictionary.dict"
path = f"../resources/{in_file}"
dictionary = corpora.Dictionary.load(path)


# In[3]:


num_topics = [
    10, 100, 200, 300, 400, 
    500, 1000, 1500, 2000, 
    2500, 3000, 3500, 4000, 
    4500, 5000, 5500, 6065
]

"""
Apply transformation to tf-idf real-valued representation' to
the whole corpus.
"""
docs_bow = [dictionary.doc2bow(doc) for doc in docs]
tfidf_model = models.TfidfModel(docs_bow)
docs_tfidf = tfidf_model[docs_bow]

queries_bow = [dictionary.doc2bow(query) for query in queries]
queries_tfidf = tfidf_model[queries_bow]

for k in num_topics:    
    """
    Perform vector transformation and other form req. models.
    """
    lsi_model = models.LsiModel(
        docs_tfidf,
        id2word=dictionary,
        num_topics=k
    )
    
    docs_lsi = lsi_model[docs_tfidf]
    index = similarities.MatrixSimilarity(docs_lsi)
    queries_lsi = lsi_model[queries_tfidf]
    
    rank_matrix = np.empty([len(queries), len(docs)])
    count = 0
    for query_lsi in queries_lsi:
            sims = index[query_lsi]
            rank_matrix[count, :] = np.argsort(-1 * sims) +  1
            count += 1
            
    rank_matrix = rank_matrix.transpose()
    rank_list = []
    for i in range(len(queries)):
        rank_list.append(list(map(int, rank_matrix[:, i])))
    
    # Save the doc. ranking matrix.
    file_name = f"ranked_doc_{k}.pickle"
    dir_path = "../hyperparameter_tests/ranked_docs"
    save_data(rank_list, file_name, dir_path)


# In[ ]:




