Introduction
This chatbot has been developed employing Natural Language Processing (NLP) techniques acquired to date. It is designed to engage in limited conversations within a specific domain by leveraging a knowledge base or extracting information directly from the internet. The chosen domain for this application is DOGS. Users may pose questions regarding dog breeds, dog nutrition, the adoption process for dogs, and related topics.

System description
This dog-focused chatbot is designed to facilitate constrained interactions with users, meticulously extracting the most suitable responses from its extensive knowledge base through the application of diverse Natural Language Processing (NLP) methodologies. If a query does not correspond with any existing entry within the knowledge base, the system initiates a real-time web search to procure pertinent information. Moreover, for inquiries that remain unresolved, the chatbot offers tailored statements or suggestions related to topics of potential relevance, thereby providing users with constructive guidance, and enhancing the overall user experience.

NLP techniques incorporated
Upon receiving a query from the user, the system is engineered to identify and present the most pertinent response available. In order to ascertain the response that most closely aligns with the user's inquiry from the knowledge base, the system employs the following techniques:

1.	Part-of-Speech tagging
Overview: A core NLP method that assigns categories (e.g., noun, verb, adjective) to words in text. My chatbot employs this technique mainly to isolate nouns from both user queries and knowledge base entries.

Application: 
a.	Noun Extraction: The get_nouns_from_pos(text) function processes input (queries or facts) through an NLP model to tokenize text and filter out NOUNS for further ananlyis. 
b.	Similarity Determination: The find_most_similar_pos_tagging(user_query) function extracts nouns from user queries and knowledge base facts, comparing them to identify overlaps. A predefined noun overlap threshold is essential for a fact to qualify as a potential response, ensuring relevance.
c.	Ranking and Selection: Facts are ranked by noun overlap magnitude, with the top-ranking fact deemed the most suitable response. The match from the ranked list based on user’s likes are given more preference. If there is nothing common, then the first result from the sorted list is displayed. 


2.	Cosine similarity
Overview: A key NLP metric for assessing the similarity between two documents regardless of size. This chatbot employs cosine similarity to compare user queries with existing facts.

Application:
a.	Similarity Calculation: Utilizes TF-IDF vectors through the calculate_similarity function to quantify similarity between a user query and facts.
b.	Threshold-Based Matching: The get_cosine_similarity function computes similarity scores, applying a threshold to filter for relevance.
c.	Ranking: Facts are prioritized by their similarity score, selecting either matches aligned with user preferences or the highest scoring fact.

3.	Dependency Parsing
Overview: An NLP technique that analyzes sentence structure to identify grammatical relationships between words. Used in our chatbot to break down both user queries and knowledge base facts into entities and their relational dynamics.

Application: 
a.	Extraction: Utilizing get_entities_and_relations_from_dependency_parsing, the system identifies crucial entities (nouns, subjects, objects)  and their interrelations (adjectives, modifiers, prepositions), within texts.
b.	Matching: In find_most_similar_dependency_parsing the system will match user queries with facts by evaluating grammatical and semantic overlaps, with a predefined threshold for relevance.
c.	Ranking: facts are ranked based on the overlap in entities and relations. The selecting the most contextually aligned fact as the response.


4.	LDA Topic modelling 
Overview:  LDA is a type of Natural Language Processing (NLP) technique used for topic modeling, which allows the extraction of topics from large volumes of text by grouping similar words into topics. In our chatbot, LDA similarity is utilized to understand the underlying topics within user queries and compare them with the topics derived from the knowledge base facts.

LDA Model Construction 
The process of constructing the Latent Dirichlet Allocation (LDA) model involves several steps aimed at understanding and categorizing the textual content into distinct topics:

a.	Data Collection and Preprocessing: Utilizing read_all_files, text is extracted and initially cleaned from specified directories. WordNetLemmatizer aids in standardizing and refining text for consistency.
b.	Dictionary and Corpus creation: From this processed text, a dictionary is formed to associate unique words with IDs, and documents are converted into a bag-of-words format, setting the stage for analysis.
c.	LDA Training and Saving: Through create_and_save_lda_model, the LDA model learns from the corpus to delineate a predetermined number of topics, enhancing its topic recognition capability. After training, the topics are evaluated for coherent representation, and the refined LDA model along with its dictionary are saved.

Application:
a.	Topic Extraction: Through get_topics_from_text, the system processes the text (after cleaning and preprocessing) to determine its significant topics by constructing a bag-of-words (BoW) and applying the LDA model to identify the distribution of topics.
b.	Threshold-Based Topic Matching: In find_most_similar_topic_modelling, the user's query topics are compared to predefined topics in the knowledge base. A relevance threshold ensures only significantly matching topics influence the response selection.
c.	Best Match Identification: The method selects the most relevant topic and its associated facts based on the highest overlap and adherence to the LDA threshold, ensuring the response is both topical and contextually relevant to the user's inquiry.


5.	NER Algorithm 
Overview: 
The Named Entity Recognition (NER) enhance text comprehension and relevance in NLP.  It categories key text elements like names, places, organizations and dates.

 Application:
a.	Entity Extraction: In the get_lesk_similarity function, NER is first applied to the user's query to isolate named entities and also for facts within the knowledge base.
b.	Semantic Matching: the system finds overlaps between entities identified in the user query and those within knowledge base facts. These facts are gathered as candidate facts.
c.	Response Determination: These candidate facts are prioritized by their score, selecting either matches aligned with user preferences or the highest scoring fact.
