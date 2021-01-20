#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *
import json
from gensim import corpora


# In[2]:


DIR_NAMES= [
    "output", "resources", "hyperparameter_tests"
]

for dir_name in DIR_NAMES:
    dir_path = f"../{dir_name}"
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


# In[3]:


"""
Cleanup (pre-process) the text.
"""
# Read documents.
docs_json = json.load(open("../Cranfield Dataset/cranfield/cran_docs.json", 'r'))[:]
doc_ids = [item["id"] for item in docs_json]
docs = [item["body"] for item in docs_json]

# Read queries.
queries_json = json.load(open("../Cranfield Dataset/cranfield/cran_queries.json", 'r'))[:]
query_ids =  [item["query number"] for item in queries_json]
queries = [item["query"] for item in queries_json]

# Clean the documents and queries.
cleaned_docs = text_cleanup(docs, saving=True, file_name="cleaned_docs.pickle")
cleaned_queries = text_cleanup(queries, saving=True, file_name="cleaned_queries.pickle")


# In[4]:


"""
Build a dictionary from the list of documents.
"""	
dictionary = corpora.Dictionary(cleaned_docs)
dictionary.save("../resources/dictionary.dict")


# In[ ]:




