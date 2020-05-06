from tokenization import Tokenization
from inflection_reduction import InflectionReduction
from stopword_removal import StopwordRemoval
from information_retrieval import InformationRetrieval
from evaluation import Evaluation

import argparse
import json
import matplotlib.pyplot as plt

class SearchEngine:

	def __init__(self, args):
		self.args = args
		self.tokenizer = Tokenization()
		self.inflection_reducer = InflectionReduction()
		self.stopword_remover = StopwordRemoval()
		self.information_retriever = InformationRetrieval(self.args.out_folder)
		self.evaluator = Evaluation()


	def tokenize(self, text):
		"""
		Return the required tokenizer.
		"""
		return self.tokenizer.tokenize(text)
	
	
	def reduce_inflection(self, text):
		"""
		Return the required tokenizer.
		"""
		return self.inflection_reducer.reduce(text)
	
	
	def remove_stopwords(self, text):
		"""
		Return the required stopword remover.
		"""
		return self.stopword_remover.from_list(text)
	
	
	def execute_ir_tasks(self, docs, queries):
		"""
		Perform tasks related to oinformation retrieval.
		"""
		return self.information_retriever.retrieval_tasks(docs, queries)
	

	def preprocess_queries(self, queries):
		"""
		Preprocessing the queries involves-
		1. Tokenizing the queries.
		2. Lemmatize the tokens.
		3. Removing the stopwords.
		"""
		# Tokenize queries.
		tokenized_queries = []
		for query in queries:
			tokenized_query = self.tokenize(query)
			tokenized_queries.append(tokenized_query)
		json.dump( \
			tokenized_queries, \
			open(self.args.out_folder + "tokenized_queries.txt", "w") \
		)

		# Stem/Lemmatize queries.
		reduced_queries = []
		for query in tokenized_queries:
			reduced_query = self.reduce_inflection(query)
			reduced_queries.append(reduced_query)
		json.dump( \
			reduced_queries, \
			open(self.args.out_folder + "reduced_queries.txt", "w") \
		)

		# Remove stopwords from queries.
		stopword_removed_queries = []
		for query in reduced_queries:
			stopword_removed_query = self.remove_stopwords(query)
			stopword_removed_queries.append(stopword_removed_query)
		json.dump( \
			stopword_removed_queries, \
			open(self.args.out_folder + "stopword_removed_queries.txt", "w") \
		)
		
		preprocessed_queries = stopword_removed_queries
		return preprocessed_queries


	def preprocess_docs(self, docs):
		"""
		Preprocessing the documents involves-
		1. Tokenizing the documents.
		2. Lemmatize the tokens.
		3. Removing the stopwords.
		"""
		# Tokenize documents.
		tokenized_docs = []
		for doc in docs:
			tokenized_doc = self.tokenize(doc)
			tokenized_docs.append(tokenized_doc)
		json.dump( \
			tokenized_docs, \
			open(self.args.out_folder + "tokenized_docs.txt", "w") \
		)

		# Stem/Lemmatize documents.
		reduced_docs = []
		for doc in tokenized_docs:
			reduced_doc = self.reduce_inflection(doc)
			reduced_docs.append(reduced_doc)
		json.dump( \
			reduced_docs, \
			open(self.args.out_folder + "reduced_docs.txt", "w") \
		)

		# Remove stopwords from documents.
		stopword_removed_docs = []
		for doc in reduced_docs:
			stopword_removed_doc = self.remove_stopwords(doc)
			stopword_removed_docs.append(stopword_removed_doc)
		json.dump( \
			stopword_removed_docs, \
			open(self.args.out_folder + "stopword_removed_docs.txt", "w") \
		)
		
		preprocessed_docs = stopword_removed_docs
		return preprocessed_docs


	def evaluate_dataset(self):
		"""
		Evaluate the document-query relevances for all document query pairs.
		"""
		# Read queries.
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
					[item["query"] for item in queries_json]

		# Process queries.
		processed_queries = self.preprocess_queries(queries)

		# Read documents.
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]#[:50]
		doc_ids, docs = [item["id"] for item in docs_json], \
					[item["body"] for item in docs_json]

		# Process documents.
		processed_docs = self.preprocess_docs(docs)

		# Perform information retrieval.
		doc_IDs_ordered = \
			self.execute_ir_tasks(processed_docs,processed_queries)		
		
		# Read relevance judements
		qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], [] 
		for k in range(1, 11):
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			precisions.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered, query_ids, qrels, k)
			recalls.append(recall)
			fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, query_ids, qrels, k)
			fscores.append(fscore)
			print("Precision, Recall and F-score @ " +  
				str(k) + " : " + str(precision) + ", " + str(recall) + 
				", " + str(fscore))
			MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			MAPs.append(MAP)
			nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, k)
			nDCGs.append(nDCG)
			print("MAP, nDCG @ " +  
				str(k) + " : " + str(MAP) + ", " + str(nDCG))

		# Plot the metrics and save plot 
		plt.plot(range(1, 11), precisions, label="Precision")
		plt.plot(range(1, 11), recalls, label="Recall")
		plt.plot(range(1, 11), fscores, label="F-Score")
		plt.plot(range(1, 11), MAPs, label="MAP")
		plt.plot(range(1, 11), nDCGs, label="nDCG")
		plt.legend()
		plt.title("Evaluation Metrics - Cranfield Dataset")
		plt.xlabel("k")
		# Own Code
		#plt.show()
		plt.savefig(args.out_folder + "eval_plot.png")
		
		# Remaning code will be added later.


if __name__ == "__main__":

	# Create an argument parser.
	parser = argparse.ArgumentParser(description="main.py")

	# Tunable parameters as external arguments.
	parser.add_argument("-dataset", default="carnfield/", help="Path to output folder")
	parser.add_argument("-out_folder", default="output/", help="Path to output folder")
	parser.add_argument("-custom", action="store_true", help="Take custom query as input")
	
	# Parse the input arguments.
	args = parser.parse_args()

	# Create an instance of the Search Engine.
	search_engine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset.
	if args.custom:
		search_engine.handle_custom_query()
	else:
		search_engine.evaluate_dataset()

