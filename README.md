# Information_Retrieval_System
In this project, we build a simple search engine that retrieves appropriate documents corresponding to a given query by considering the semantic relatedness between the texts of the documents and the query. We study how to represent documents and queries as vectors in various representational spaces so that the documents most similar to any partcular query can be retrieved using an appropriate similarity measure. We also test the search engine's performance on a benchmark document collection and set of questions and present the results for the same.

## Some Basics of Informattion Retrieval
The goal of an IR system is to retrieve documents with information relevant to a user's information need and, consequently, help a user complete a task.

<div align="center">
    <p>
        <img src="images/classic_search_model.png" alt="Classic Search Model" width=60%, height=80%>
    </p>  
</div>

Assume that a user has a task they want to perform. For example, let us assume that a user wants to buy a mask as a preventive measure against the Covid-19 disease.
To achieve this task, the user needs various information like the
different types of masks available in the market, the standard of mask prescribed by the health officials, the stores currently selling the prescribed masks
etc. We refer to this as the Information Need of the user, and it is this information need that we use to assess an IR system. Hence, we type a query in the search box and let the IR system retrieve documents that are relevant to the query from the collections (i.e., a set of documents) and display them as the results. If the user is not satisfied with the results, then they may refine the query and again search documents concerning the new query to get desired results.



## Instructions
Please follow the steps given below-

1. Copy 'Cranfield' (which contains training documents and queries for this project) and 'source_code' to the same directory (if not already done).
2. **TRAINING-** Run the scripts in the following order- <br>
	(a) Execute- preprocess_docs_and_queries.py <br>
	(b) Execute- ranked_list_lsi_models_different_topics.py <br>
	(c) Execute- performance_comparison_lsi_models_different_topics.py <br>
	(d) Execute- vsm_models_performance_comparison.py <br>
3. **TESTING-** Let the query for which yu want to retrieve documents be "xyz". <br>
	(a) Execute- retrieve_docs_for_input_query.py -custom "xyz" <br>
	
**Created by-** <br>
	*Debargha Bhattacharjee* <br>
	*CS19S028, MS Scholar* <br>
	*Department of Computer Science and Engineering* <br>
	*IIT Madras* <br>
	*CS6370 Natural Language Processing Course Project* <br>
	
