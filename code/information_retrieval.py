from util import *

# Add your import statements here.
import numpy as np
from gensim import corpora
from gensim import models
from gensim import similarities

class InformationRetrieval():

	def __init__(self, dir_path):
		self.out_dir = dir_path
		self.lsi_model = None
		self.dictionary = None
		"""
		CUSTOM ATTRIBUTE INSERTED HERE.
		"""

	def build_lsi_model(self, docs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		"""
		Build a dictionary from the list of documents.
		"""	
		dictionary = corpora.Dictionary(docs)
		self.dictionary = dictionary
		dictionary.save(self.out_dir + "dictionary.dict")

		"""
		Build a corpus from the dictionary.
		"""
		corpus = [dictionary.doc2bow(doc) for doc in docs]
		corpora.MmCorpus.serialize(self.out_dir + "corpus.mm", corpus)

		"""
		Appy 'transformation to tfidf real-valued representation' 
		to the whole corpus.
		"""
		tfidf = models.TfidfModel(corpus)
		corpus_tfidf = tfidf[corpus]
		
		"""
		Perform vector transformation and other form required models.
		"""
		lsi_model =models.LsiModel(
				corpus_tfidf, 
				id2word=dictionary, 
				num_topics=300
		)
		self.lsi_model = lsi_model 



	def rank(self, docs, queries):
		dictionary = self.dictionary

		docs_bow = [dictionary.doc2bow(doc) for doc in docs]
		tfidf = models.TfidfModel(docs_bow)
		docs_tfidf = tfidf[docs_bow]
		docs_lsi =  self.lsi_model[docs_tfidf]

		index = similarities.MatrixSimilarity(docs_lsi)

		queries_bow = [dictionary.doc2bow(query) for query in queries]
		tfidf = models.TfidfModel(queries_bow)
		queries_tfidf = tfidf[queries_bow]
		queries_lsi = self.lsi_model[queries_tfidf]
		

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
		return rank_list	
		
		

	def retrieval_tasks(self, docs, queries):
		"""
		Perform tasks related to Information Retrieval such as-
		1. Build dictionary.
		2. Build corpus.
		"""
		self.build_lsi_model(docs)
		rank_list = self.rank(docs, queries)
		return rank_list

	
	
		
		
		
