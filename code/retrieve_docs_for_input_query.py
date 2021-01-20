#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *
from evaluation import Evaluation
import numpy as np
from gensim import corpora
from gensim import models
from gensim import similarities
import argparse
import json
import matplotlib.pyplot as plt


# In[2]:


# # Create an argument parser.
parser = argparse.ArgumentParser(description="retrieve_docs_for_input_query.py")
parser.add_argument("-custom", help="Enter custom query.")
# # Parse the input arguments.
args = parser.parse_args()


# In[3]:


"""
Retrieve documents for a custom query.
"""
# Read documents.
docs_dir = "../Cranfield Dataset/cranfield/cran_docs.json"
docs_json = json.load(open(docs_dir, 'r'))[:]
doc_ids = [item["id"] for item in docs_json]
docs = [item["body"] for item in docs_json]
        
custom_query = [str(args.custom).lower()]
print(f"Entered query: {custom_query[0]}")
preprocessed_custom_query = text_cleanup(custom_query)

# Load the preprocessed docs.
processed_docs = load_data("../output/cleaned_docs.pickle")

# Retrieve saved models to perform IR.
"""
Load saved dictionary, tf-idf model, lsi model.
"""
dictionary =     corpora.Dictionary.load("../resources/dictionary.dict")
tfidf_model =     models.TfidfModel.load("../resources/tfidf_model.tfidf")
lsi_model =     models.LsiModel.load("../resources/lsi_model.lsi")


# In[7]:


docs_bow = [dictionary.doc2bow(doc) for doc in processed_docs]
docs_tfidf = tfidf_model[docs_bow]
docs_lsi = lsi_model[docs_tfidf]

index = similarities.MatrixSimilarity(docs_lsi)

queries_bow = [dictionary.doc2bow(query) for query in preprocessed_custom_query]      
queries_tfidf = tfidf_model[queries_bow]
queries_lsi = lsi_model[queries_tfidf]

rank_matrix = np.empty([len(preprocessed_custom_query), len(docs)])
count = 0
for query_lsi in queries_lsi:
    sims = index[query_lsi]
    rank_matrix[count, :] = np.argsort(-1 * sims) +  1
    count += 1

rank_matrix = rank_matrix.transpose()
rank_list = []
for i in range(len(preprocessed_custom_query)):
    rank_list.append(list(map(int, rank_matrix[:, i])))
doc_IDs_ordered =  rank_list


# In[8]:


nb_ordered_docs = None
doc_id = None
option = "y"

while(True):
    k = input("Enter top k number of docs you want to view? (1 <= k <= 1400): ")
    try:
        if not 1 <= int(k) <= 1400:
            raise Exception()
        nb_ordered_docs = int(k)
        break
    except:
        print("Invalid k.")
        print("k should be between 1 and 1400 (both inclusive).")

while(True):
    retrieved_doc_ids = doc_IDs_ordered[0][:nb_ordered_docs]
    print(f"The IDs of the top {nb_ordered_docs} documents are as follows: ")
    print(retrieved_doc_ids)
    print()

    while(True):
        choice = input("Enter ID of document (from list) you want to view: ")
        try:
            doc_id = int(choice)
            if not doc_id in retrieved_doc_ids:
                raise Exception()
            print("==============================================================")
            print(f"Document {doc_id}")
            print("--------------------------------------------------------------")
            print(docs[doc_id-1])
            print("==============================================================")
            break
        except:
            print("Invalid ID.")
            print("Enter a valid ID from the list.")

    option = input("Want to see more documents? (y/n): ")
    option = str(option)
    if ((option=="y") or (option=="Y")):  
        while(True):
            inp_msg = "Enter '0' to view document IDs from the same " +             "list, or enter some valid integer k (1 <= k <= 1400) if you want to update list: "
            k = input(inp_msg)
            try:
                if not 0 <= int(k) <= 1400:
                    raise Exception()
                if int(k) > 0:
                    nb_ordered_docs = int(k)
                break
            except Exception:
                print("Invalid input.")
                print("Input should be a valid integer between 1 and 1400 (both inclusive).")
    else:
        break

print("Thank you for using the search engine.")
print("Hope your experience was good.")


# In[ ]:




