import os
import json
import requests
import requests

from bs4 import BeautifulSoup
from collections import Counter
from gensim import corpora, models
from googlesearch import search 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from search_engine_parser.core.exceptions import NoResultsOrTrafficError

from utils import clean_text_lda, preprocess_text, handle_unknown_query, get_greeting, dog_facts, farewells, acknowledgments, sia, get_user_input, nlp, initial_message

user_model = None
username = None
liked_terms = []

def load_lda_model(model_path="lda_model.model"):
    if os.path.exists(model_path):
        return models.LdaModel.load(model_path)
    else:
        print(f"Error: LDA model not found at {model_path}")
        return None

lda_model = load_lda_model()
dictionary = corpora.Dictionary.load('dog_lda_dict.gensim')


# ----------------- FILTER REPSOSNE BASED ON LIKES  ------------------------
def get_match_based_on_user(all_matches, best_match):
  if len(liked_terms) > 0:
    if best_match and any(term in best_match[1].lower() for term in liked_terms):
        return best_match
      
    for match in all_matches:
        if any(term in match[1].lower() for term in liked_terms):
            return match

  return best_match
  
def extract_important_terms(likes_list):
    combined_text = ' '.join(likes_list)
    terms = clean_text_lda(combined_text)
    term_frequencies = Counter(terms)
    return [term for term, freq in term_frequencies.items() if freq > 1]
  
# ----------------- COSINE SIMILARITY ------------------------
def get_cosine_similarity(user_query):
  max_similarity = 0.35
  best_match = None
  all_matches = set()
  for topic, facts in dog_facts.items():
      for fact in facts:
          similarity = calculate_similarity(user_query, fact)
          if similarity >= max_similarity:
              max_similarity = similarity
              best_match = (topic, fact)
              all_matches.add((topic, fact))
  return get_match_based_on_user(all_matches, best_match)


def calculate_similarity(user_query, doc):
    # Preprocess both user query and document text
    user_query_processed = preprocess_text(user_query)
    doc_processed = preprocess_text(doc)

    # Create a TF-IDF Vectorizer and convert texts into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    combined_texts = [user_query_processed, doc_processed]
    tfidf_vectors = vectorizer.fit_transform(combined_texts)

    # Calculate cosine similarity score between the vectors
    similarity_scores = cosine_similarity(tfidf_vectors[0:1], tfidf_vectors[1:2])
    return similarity_scores[0][0]

# ----------------- LESK and NER  ------------------------
def get_lesk_similarity(user_query):
  '''
  This method would obtain all the entitiies from the user query using NER. 
  It then performs the same on the knowledge base and finds a match.
  '''
  user_query_processed = preprocess_text(user_query)
  user_query_doc = nlp(user_query_processed)
  user_query_entities = set(ent.text for ent in user_query_doc.ents)
  
  candidate_facts = []
  for topic, facts in dog_facts.items():
    for fact in facts:
        fact_doc = nlp(fact)
        fact_entities = set(ent.text for ent in fact_doc.ents)
        if user_query_entities.intersection(fact_entities):
            candidate_facts.append((topic, fact))
  
  # best_lesk_fact = max(lesk_scores, key=lesk_scores.get, default=None)
  if len(candidate_facts) == 0: 
    return None
  # return candidate_facts
  if candidate_facts: return get_match_based_on_user(candidate_facts, candidate_facts[0])

# ----------------- POS TAGGING  ------------------------
def get_nouns_from_pos(text):
  doc = nlp(text)
  nouns = [token.text for token in doc if token.pos_ == "NOUN"]
  return nouns

def find_most_similar_pos_tagging(user_query):
  threshold = 2
  user_nouns = get_nouns_from_pos(user_query)
  candidate_facts = []
  for topic, facts in dog_facts.items():
    for fact in facts:
      fact_nouns = get_nouns_from_pos(fact)
      overlap = len(set(user_nouns).intersection(set(fact_nouns)))
      if overlap >= threshold:
        candidate_facts.append((topic, fact, overlap))

  # Rank facts based on noun overlap (you can add other similarity measures)
  ranked_facts = sorted(candidate_facts, key=lambda x: x[2], reverse=True)
  if ranked_facts: return get_match_based_on_user(ranked_facts, ranked_facts[0])
  return None

# ----------------- DEPENDENCY GRAPH  ------------------------
def get_entities_and_relations_from_dependency_parsing(text):
    text = preprocess_text(text)
    doc = nlp(text)
    entities = []
    relations = []
    for token in doc:
        if token.dep_ in ["nsubj", "obj", "ROOT"]:  # Focus on key entities
            entities.append(token.text.lower())
        if token.dep_ in ["amod", "nmod", "prep"]:  # Extract common relationships
            relations.append(token.text.lower())
    return entities, relations

def find_most_similar_dependency_parsing(user_query):
    threshold = 1
    user_entities, user_relations = get_entities_and_relations_from_dependency_parsing(user_query)
    # print("USER ENTITIES: ", user_entities, "\n")
    candidate_facts = []
    for topic, facts in dog_facts.items():
        for fact in facts:
            fact_entities, fact_relations = get_entities_and_relations_from_dependency_parsing(fact)
            # print(fact_entities)
            overlap_entities = len(set(user_entities).intersection(set(fact_entities)))
            overlap_relations = len(set(user_relations).intersection(set(fact_relations)))
            similarity_score = overlap_entities + overlap_relations
            if similarity_score >= threshold:
              candidate_facts.append((topic, fact, similarity_score))

    # Rank facts based on entity and relation overlap
    ranked_facts = sorted(candidate_facts, key=lambda x: x[2], reverse=True)
    if ranked_facts: return get_match_based_on_user(ranked_facts, ranked_facts[0])
    return None
  
  
# ----------------- LDA SIMILARITY ------------------------
def get_topics_from_text(text):
    preprocessed_text = clean_text_lda(text)
    bow = dictionary.doc2bow(preprocessed_text)
    user_topics = lda_model.get_document_topics(bow)
    sorted_topics = sorted(user_topics, key=lambda x: x[1], reverse=True)
    return sorted_topics

def find_most_similar_topic_modelling(user_query):
    threshold = 0.6
    user_topics = get_topics_from_text(user_query)
    # print(user_topics)
    # print(lda_model.show_topic(user_topics[0][0]))
    best_match = []
    if user_topics:  # Check if the list is not empty
        most_relevant_topic_id = user_topics[0][0]  # Get relevant topic
        if user_topics[0][1] >= threshold: # this is the threshold for LDA
          topic_terms = lda_model.show_topic(most_relevant_topic_id)
          for word, probability in topic_terms:
            # print(word, probability)
            for key,facts in dog_facts.items():
              if word == key:
                best_match = (word, dog_facts[word])
                threshold = max(threshold, user_topics[0][1])
        else:
          return None
    if best_match: return best_match[:2]
    return None


# ----------------- Web Lookup ------------------------
def get_first_web_link_summary(query):
    first_url = None
    try:
        urls = [link for link in search(query, num=10, stop=10, pause=2) if not any(substring in link for substring in ['https://www.youtube.com', '.jpg', '.png', '.gif', 'image', 'img'])]
        
        if not urls:
            return None
        
        first_url = urls[0]        

        response = requests.get(first_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        paragraphs = soup.find_all('p')
        summary = ' '.join([para.text for para in paragraphs[:2]]) if paragraphs else 'No text available'
        
        return first_url, summary

    except requests.exceptions.RequestException as e:
        return first_url, "Failed to retrieve content: " + str(e)


# ----------------- MAIN function ------------------------
def find_most_similar(user_query):
    # Initialize all match variables to None
    best_match_cosine = None
    best_match_graph = None
    best_match_topic = None
    best_match_pos = None
    best_match_lesk = None

    # Method 3: POS Tagging
    best_match_pos = find_most_similar_pos_tagging(user_query)
    if best_match_pos is not None:
        return best_match_pos
      
    # METHOD 1: COSINE SIMILARITY
    best_match_cosine = get_cosine_similarity(user_query)
    if best_match_cosine is not None:
        return best_match_cosine

    # Method 2: LDA Topic modelling
    best_match_topic = find_most_similar_topic_modelling(user_query)
    if best_match_topic is not None:
        return best_match_topic

    # Method 4: Dependency graphs, make sure the graph score is not 0
    best_match_graph = find_most_similar_dependency_parsing(user_query)
    if best_match_graph is not None:
        return best_match_graph

    # METHOD 5: Using NER to get the entities. Sometimes the user query might be NULL
    best_match_lesk = get_lesk_similarity(user_query)
    if best_match_lesk is not None:
        return best_match_lesk
      
    return None

# ----------------- USER MODEL ------------------------
def get_or_create_user_model(username):
    directory = "user_models"
    file_path = os.path.join(directory, f"{username}.json")
    if not os.path.exists(directory):
        os.makedirs(directory)
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            user_model = json.load(file)
            print(f"Welcome back {user_model.get('name', 'there')}!")  # Welcome back message
            return user_model
    else:
        name = input("I see it's your first time here. What's your name? ")
        return {"name": name, "username": username, "likes": [], "dislikes": []}

def save_user_model(user_model):
    directory = "user_models"
    file_path = os.path.join(directory, f"{user_model['username']}.json")
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(user_model, file, ensure_ascii=False, indent=4)

def add_to_user_preferences(user_model, message):
    sentiment_score = sia.polarity_scores(message)
    if sentiment_score['compound'] > 0.05:  # Positive sentiment
        user_model['likes'].append(message)
    elif sentiment_score['compound'] < -0.05:  # Negative sentiment
        user_model['dislikes'].append(message)
    

def main():
    print(initial_message)
    username = input("Enter your username to start: ")
    user_model = get_or_create_user_model(username.lower())  # Convert username to lowercase for consistency
    liked_terms = extract_important_terms(user_model['likes'])
    print(get_greeting())

    while True:
        user_input = get_user_input()
        if len(user_input) < 1:
          print("Please enter a valid question or statement.")
          break
        add_to_user_preferences(user_model, user_input)  
        if user_input.lower() == "quit": 
            save_user_model(user_model)
            break
        
        if user_input.lower() in acknowledgments:
            response = "Hope you were satisfied with my answer."
            print(response)
            print("We can continue further. Ask me anything about Dogs.")
            continue

        if user_input.lower() in farewells:
            response = "Hope you were satisfied with my answers. Goodbye!"
            print(response)
            break

        response = find_most_similar(user_input)
        if response is not None and len(response) >= 2:
            print(f"Here's what I found about the topic '{response[0]}': {response[1]}")
        else:
            # Method 6: Optional live web lookup
            print("Live web lookup is happening... hang in there!")
            try:
              web_lookup_result = get_first_web_link_summary(user_input)
              if web_lookup_result is not None:
                  url, summary = web_lookup_result
                  print(f"I found something on the web that might help: {url} - Here's a summary: {summary}")
            except NoResultsOrTrafficError:
                print("Search failed due to no results or unusual traffic. Please try again later.")
            print("\n")
            print(handle_unknown_query())

        save_user_model(user_model)


if __name__ == "__main__":
  main()
  


# LESK and NER 


# if not ner entities (SPacy)
#     within that lesk with potential topic 
# then dependency parsing 
# if not then POS (noun)

