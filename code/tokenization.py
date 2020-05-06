from util import *

# Add your import statements here
from gensim.utils import simple_preprocess


class Tokenization():

	def tokenize(self, text):

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
