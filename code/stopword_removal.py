from util import *

# Add your import statements here
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize



class StopwordRemoval():

	def from_list(self, text):
		"""
		Stopwords removal.
		
		Parameters
		----------
		arg1 : list
		A list of lemmatized tokens.
		
		Returns
		-------
		list
		A new list with stopwords removed. 
		"""
		
		stopword_removed_text = None
		
		#Fill in code here

		stop_words = set(stopwords.words('english'))
		cleaned_text = [
				tokens
				for tokens in text
				if tokens not in stop_words
		]
		stopword_removed_text = cleaned_text
		
		return stopword_removed_text




	
