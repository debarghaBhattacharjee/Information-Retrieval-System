from util import *

# Add your import statements here
from nltk.stem import WordNetLemmatizer
#from gensim.utils import lemmatize as lm



class InflectionReduction:
	def reduce(self, text):
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


