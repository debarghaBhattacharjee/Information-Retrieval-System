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

import matplotlib.pyplot as plt
from evaluation import Evaluation


# In[2]:


# Load the preprocessed docs.
docs = load_data("../output/cleaned_docs.pickle")

# Load the preprocessed queries.
queries = load_data("../output/cleaned_queries.pickle")

dict_path = "../resources/dictionary.dict"
dictionary = corpora.Dictionary.load(dict_path)


# In[3]:


"""
Ranking documents using the tf-idf model.
"""
docs_bow = [dictionary.doc2bow(doc) for doc in docs]
tfidf_model = models.TfidfModel(docs_bow)
tfidf_model.save("../resources/tfidf_model.tfidf")
docs_tfidf = tfidf_model[docs_bow]

index_tfidf = similarities.MatrixSimilarity(docs_tfidf)

queries_bow = [dictionary.doc2bow(query) for query in queries]
queries_tfidf = tfidf_model[queries_bow]

rank_matrix_tfidf = np.empty([len(queries), len(docs)])
count = 0
sims_tfidf = []

for query_tfidf in queries_tfidf:
    sims = index_tfidf[query_tfidf]
    sims_tfidf.append(sims)
    rank_matrix_tfidf[count, :] = np.argsort(-1 * sims) +  1
    count += 1

rank_matrix_tfidf = rank_matrix_tfidf.transpose()
rank_list_tfidf = []
for i in range(len(queries)):
    rank_list_tfidf.append(list(map(int, rank_matrix_tfidf[:, i])))

# Save the tf-idf doc. ranking matrix.
file_name = "ranked_list_tfidf"
dir_path = "../hyperparameter_tests/compare_vsm"
save_data(rank_list_tfidf, file_name, dir_path)


# In[4]:


"""
Ranking different documents using the LSI model.
Build the LSI model using the tf-idf matrix already created.
"""

"""
Create the lsi model.
"""
lsi_model = models.LsiModel(
    docs_tfidf,
    id2word=dictionary,
    num_topics=300
)
lsi_model.save("../resources/lsi_model.lsi")

docs_lsi = lsi_model[docs_tfidf]
index = similarities.MatrixSimilarity(docs_lsi)
queries_lsi = lsi_model[queries_tfidf]

rank_matrix_lsi = np.empty([len(queries), len(docs)])
count = 0

sims_lsi = []

for query_lsi in queries_lsi:
    sims = index[query_lsi]
    sims_lsi.append(sims)
    rank_matrix_lsi[count, :] = np.argsort(-1 * sims) +  1
    count += 1
    
rank_matrix_lsi = rank_matrix_lsi.transpose()
rank_list_lsi = []
for i in range(len(queries)):
    rank_list_lsi.append(list(map(int, rank_matrix_lsi[:, i])))

# Save the LSI doc. ranking matrix.
file_name = "ranked_list_lsi"
dir_path = "../hyperparameter_tests/compare_vsm"
save_data(rank_list_lsi, file_name, dir_path)


# In[5]:


"""
Ranking documents using an hybrid of tf-idf and lsi model.
"""
rho = 0.60

rank_matrix_hybrid = np.empty([len(queries), len(docs)])
count = 0

for i in range(len(queries)):
    sims =         ((rho) * sims_lsi[i]) + ((1 - rho) * sims_tfidf[i])
    rank_matrix_hybrid[count, :] = np.argsort(-1 * sims) +  1
    count += 1

rank_matrix_hybrid = rank_matrix_hybrid.transpose()
rank_list_hybrid = []
for i in range(len(queries)):
    rank_list_hybrid.append(list(map(int, rank_matrix_hybrid[:, i])))

# Save the hybrid doc. ranking matrix.
file_name = "ranked_list_hybrid"
dir_path = "../hyperparameter_tests/compare_vsm"
save_data(rank_list_hybrid, file_name, dir_path)


# In[6]:


"""
Evaluating performance of the IR system w.r.t. the different models.
"""
evaluator = Evaluation()

"""
Load cran_qrels.json to get the information about the relevant docs
corresponding to each query. Also, load cran_queries.json to get the
the query_ids list.
"""
path = "../Cranfield Dataset/cranfield/cran_qrels.json"
json_in = open(path, "r")
qrels = json.load(json_in)[:]

path = "../Cranfield Dataset/cranfield/cran_queries.json"
json_in = open(path, "r")
queries_json = json.load(json_in)[:]
query_ids = [item["query number"] for item in queries_json]

doc_ids_ordered_tfidf = rank_list_tfidf
doc_ids_ordered_lsi = rank_list_lsi
doc_ids_ordered_hybrid = rank_list_hybrid

# Calculate precision, recall, f-score, 
# MAP and nDCG for k = 1 to 10
mean_avg_precisions_tfidf = []
mean_avg_precisions_lsi = []
mean_avg_precisions_hybrid = []

for k in range(1, 11):
    mean_avg_precision_tfidf = evaluator.meanAveragePrecision(
        doc_ids_ordered_tfidf, query_ids, qrels, k
    )
    mean_avg_precision_lsi = evaluator.meanAveragePrecision(
        doc_ids_ordered_lsi, query_ids, qrels, k
    )
    mean_avg_precision_hybrid = evaluator.meanAveragePrecision(
        doc_ids_ordered_hybrid, query_ids, qrels, k
    )
    mean_avg_precisions_tfidf.append(mean_avg_precision_tfidf)
    mean_avg_precisions_lsi.append(mean_avg_precision_lsi)
    mean_avg_precisions_hybrid.append(mean_avg_precision_hybrid)


# In[7]:


"""
Graphical representation of the perfromance of the IR system w.r.t.
the different models.
"""
plt.figure(figsize=(10, 6))
    
plt.xlabel("k")
plt.ylabel("Mean. avg. precision")
plt.title(
    "Performance of IR system for tf-df,lsi model and hybrid model."
)
plt.plot(
    range(1, 11), 
    mean_avg_precisions_tfidf,
    label="tf-idf model"
)
plt.plot(
    range(1, 11), 
    mean_avg_precisions_lsi,
    label="lsi model"
)
plt.plot(
    range(1, 11), 
    mean_avg_precisions_hybrid,
    label="hybrid model"
)
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.savefig("../output/test-vsm_models_performance_comparison_evaluation_metric-map_upto_rank_10.pdf")
plt.show()


# In[8]:


# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], [] 
for k in range(1, 11):
    precision = evaluator.meanPrecision(
        doc_ids_ordered_lsi, query_ids, qrels, k)
    precisions.append(precision)
    recall = evaluator.meanRecall(
        doc_ids_ordered_lsi, query_ids, qrels, k)
    recalls.append(recall)
    fscore = evaluator.meanFscore(
        doc_ids_ordered_lsi, query_ids, qrels, k)
    fscores.append(fscore)
    print("Precision, Recall and F-score @ " +  
        str(k) + " : " + str(precision) + ", " + str(recall) + 
        ", " + str(fscore))
    MAP = evaluator.meanAveragePrecision(
        doc_ids_ordered_lsi, query_ids, qrels, k)
    MAPs.append(MAP)
    nDCG = evaluator.meanNDCG(
        doc_ids_ordered_lsi, query_ids, qrels, k)
    nDCGs.append(nDCG)
    print("MAP, nDCG @ " +  
        str(k) + " : " + str(MAP) + ", " + str(nDCG))

# Plot the metrics and save plot 
# Plot the metrics and save plot 
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), precisions, label="Precision")
plt.plot(range(1, 11), recalls, label="Recall")
plt.plot(range(1, 11), fscores, label="F-Score")
plt.plot(range(1, 11), MAPs, label="MAP")
plt.plot(range(1, 11), nDCGs, label="nDCG")
plt.legend()
plt.title("Evaluation Metrics - Cranfield Dataset")
plt.xlabel("Rank")
plt.savefig("../output/eval_plot.pdf")
plt.show()


# In[ ]:




