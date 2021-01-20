#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *
import pickle
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from evaluation import Evaluation


# In[2]:


evaluator = Evaluation()

num_topics = [
    10, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500,
         3000, 3500, 4000, 4500, 5000, 5500, 6065
]

# Load the preprocessed docs.
path = "../Cranfield Dataset/cranfield/cran_qrels.json"
json_in = open(path, "r")
qrels = json.load(json_in)[:]

path = "../Cranfield Dataset/cranfield/cran_queries.json"
json_in = open(path, "r")
queries_json = json.load(json_in)[:]
query_ids = [item["query number"] for item in queries_json]

maps_compare = []
for n in num_topics:
    data_path = f"../hyperparameter_tests/ranked_docs/ranked_doc_{n}.pickle"
    doc_IDs_ordered = load_data(data_path)
    
    # Calculate precision, recall, f-score, 
    # MAP and nDCG for k = 1 to 10
    mean_avg_precisions = []
    for k in range(1, 11):
        mean_avg_precision = evaluator.meanAveragePrecision(
            doc_IDs_ordered, query_ids, qrels, k
        )
        mean_avg_precisions.append(mean_avg_precision)
    maps_compare.append(mean_avg_precisions)


# In[3]:


map_at_10 = [] 
for i in range(len(num_topics)):
    map_at_10.append(maps_compare[i][-1])

plt.figure(figsize=(10, 6))
print("Min. Avg. Precision @ 10")
print("-----------------------------------------")
print(map_at_10)
print("")
    
plt.xlabel("Dimension")
plt.ylabel("Min. avg. precision @ rank 10")
plt.title(
    "Mean average precision @ rank 10 for different dimensionality for LSI applications."
)
plt.plot(
    num_topics,
    map_at_10,
)
plt.savefig("../output/test-lsi_models_nb_topics_test_evaluation_metric-map_at_rank_10.pdf")
plt.show()


# In[4]:


num_topics = [
    10, 100, 300, 500, 1000, 2000
]
plt.figure(figsize=(10, 6))
plt.xlabel("Rank")
plt.ylabel("Mean Avg. Precision")
plt.title(
    "Variation in mean average precision with change in dimensionality for LSI applications."
)
for i in range(len(num_topics)):
    plt.plot(
        range(1, 11), 
        maps_compare[i], 
        label="@Dimension {}".format(num_topics[i])
    )
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
plt.savefig("../output/test-lsi_models_performance_comparison_evaluation_metric-map_upto_rank_10.pdf")
plt.show()


# In[ ]:




