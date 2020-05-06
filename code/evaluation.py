from util import *

# Add your import statements here
import numpy as np
import math



class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		relevant_docs_list = []
		precision_query_sample_list = []
		precision_query_avg_list = []
		rel_count = 0
		avg_precision = 0
		meanPrecision = -1
	
		for i in range(len(query_ids)):
			current_query = []
			for j in range(len(qrels)):
				if (int(qrels[j]["query_num"]) == i+1):
					current_query.append(int(qrels[j]["id"]))
			relevant_docs_list.append(current_query)

		precision_query_avg_list = []
		num_queries = len(query_ids)
		for i in range(num_queries):
			rel_count = 0
			precision_query_sample_list = []
			for j in range(k):
				if doc_IDs_ordered[i][j] in relevant_docs_list[i]:
					rel_count += 1
				precision_curr = rel_count/(j+1)
				precision_query_sample_list.append(precision_curr)
			precision_query_avg_list.append(precision_query_sample_list[-1])			


		#Fill in code here
		meanPrecision = np.mean(precision_query_avg_list)
		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1
		#Fill in code here

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		relevant_docs_list = []
		recall_query_sample_list = []
		recall_query_avg_list = []
		rel_count = 0
		avg_recall = 0
	
		for i in range(len(query_ids)):
			current_query = []
			for j in range(len(qrels)):
				if (int(qrels[j]["query_num"]) == i+1):
					current_query.append(int(qrels[j]["id"]))
			relevant_docs_list.append(current_query)

		

		recall_query_avg_list = []
		num_queries = len(query_ids)
		for i in range(num_queries):
			rel_count = 0
			recall_query_sample_list = []
			total_rel_doc_i = len(relevant_docs_list[i])
			for j in range(k):
				if doc_IDs_ordered[i][j] in relevant_docs_list[i]:
					rel_count += 1
				recall_curr = rel_count/total_rel_doc_i
				recall_query_sample_list.append(recall_curr)
			
			recall_query_avg_list.append(recall_query_sample_list[-1])			


		#Fill in code here
		meanRecall = np.mean(recall_query_avg_list)		

		#Fill in code here

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1
		mean_precision = self.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
		mean_recall = self.meanRecall(doc_IDs_ordered, query_ids, qrels, k)

		meanFscore = (2 * mean_precision * mean_recall) / (mean_precision + mean_recall)
		

		#Fill in code here

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		
		relevant_docs_list = []
		relevant_docs_pos_list = []
		ndcg_query_sample_list = []
		ndcg_query_avg_list = []
		rel_count = 0
		avg_recall = 0

	
		for i in range(len(query_ids)):
			current_query_id = []
			current_query_pos = []
			for j in range(len(qrels)):
				if (int(qrels[j]["query_num"]) == i+1):
					current_query_id.append(int(qrels[j]["id"]))
					current_query_pos.append(int(qrels[j]["position"]))
			relevant_docs_list.append(current_query_id)
			relevant_docs_pos_list.append(current_query_pos)

		rndcg_query_avg_list = []
		num_queries = len(query_ids)
		curr_rel_pos_list = []

		for i in range(num_queries):
			rel_count = 0
			ndcg_curr_query_sample_list = []
			ndcg_query_sample_list = []
			total_rel_doc_i = len(relevant_docs_list[i])
			ndcg_curr_query_sum = 0
			for j in range(k):
				for j_inn in range(len(relevant_docs_list[i])):
					if doc_IDs_ordered[i][j] == relevant_docs_list[i][j_inn]:
						ndcg_curr_query_sum += \
						    relevant_docs_pos_list[i][j_inn]/math.log((j+1)+1, 2)
					break
				
			ndcg_query_avg_list.append(ndcg_curr_query_sum)			



		#Fill in code here

		meanNDCG = np.mean(ndcg_query_avg_list)

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		relevant_docs_list = []
		precision_query_sample_list = []
		precision_query_avg_list = []
		rel_count = 0
		avg_precision = 0

	
		for i in range(len(query_ids)):
			current_query = []
			for j in range(len(qrels)):
				if (int(qrels[j]["query_num"]) == i+1):
					current_query.append(int(qrels[j]["id"]))
			relevant_docs_list.append(current_query)

		

		precision_query_avg_list = []
		num_queries = len(query_ids)
		for i in range(num_queries):
			rel_count = 0
			precision_query_sample_list = []
			for j in range(k):
				if doc_IDs_ordered[i][j] in relevant_docs_list[i]:
					rel_count += 1
				precision_curr = rel_count/(j+1)
				precision_query_sample_list.append(precision_curr)
			
			precision_query_avg_list.append(np.mean(precision_query_sample_list))	


		#Fill in code here
		meanAveragePrecision = np.mean(precision_query_avg_list)
		
		return meanAveragePrecision

