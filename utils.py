
import pickle
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy  
sia = SentimentIntensityAnalyzer()
wnl = WordNetLemmatizer()

english_stopwords = set(stopwords.words('english'))
other_stopwords = ['copyright', 'could', 'would', 'should', 'butterfly', 'south', 'already', 'getting', 'someone', 'thought', 'often','butterfly', 'actually', 'clear', 'taking', 'woman', 'around']
english_stopwords = english_stopwords.union(set(other_stopwords))

greetings = ["Hi there! Would you like to know anything about dogs?", "Hello! How can I help you today?", "Welcome! What would you like to know about dogs?"]
farewells = {'bye', 'goodbye', 'cya', 'see you later'}
acknowledgments = {'ok', 'thank you', 'thanks', 'yes', 'no'}

with open("dog_facts.pickle", "rb") as f:
    dog_facts = pickle.load(f)
    
nlp = spacy.load("en_core_web_sm")  # Pre-trained NER model

initial_message = "This is a Chat Bot which can provide information on Dogs.\nThere are 5 different kinds of NLP Tehcniques which are used to return the best answer from the Knowledge base created. If no match is found a live web look up takes place and returns URL for reference.\nPlease enter 'Quit' to exit. \n\n"

# PRE PROCESSING FUNCTIONS
def clean_text_lda(text):
    '''Cleaning involves - Tokenization, removing punctuations, removing stop words and lemmatization. Return as tokens.'''
    text = text.replace('\n', '')
    content = text.lower()
    tokens = word_tokenize(content)  
    tokens = [wnl.lemmatize(t) for t in tokens if t.isalpha() and t not in english_stopwords and len(t) > 4]
    return tokens

def preprocess_text(text):
  '''Cleaning involves - Tokenization, removing punctuations, removing stop words and lemmatization. Return as STRING.'''
  wnl = WordNetLemmatizer()
  stop_words = set(stopwords.words('english'))
  tokens = [wnl.lemmatize(t) for t in text.split() if t not in stop_words and len(t) > 3]
  return " ".join(tokens)

def get_user_input():
  user_input = input("You: ")
  return user_input.strip()

def get_greeting():
  import random
  return random.choice(greetings)

def handle_unknown_query():
    sample_terms = random.sample(list(dog_facts.keys()), 2)
    replies = [
        f"Hmm, that's an interesting question! I'm still learning about dogs. Perhaps you can rephrase your question to focus on specific aspects like '{sample_terms[0]}' or '{sample_terms[1]}' related to dogs.",
        f"I can't answer that directly, but if you try asking about '{sample_terms[0]}' or '{sample_terms[1]}' related to dogs, I might be able to help.",
        "While I don't have that information, you can try searching online!",
        f"Oh, my knowledge circuits are getting overloaded! Maybe you can ask a human expert about that. I can help you with '{sample_terms[0]}' or '{sample_terms[1]}' related to dogs."
    ]
    
    # Select a random reply from the list
    index = random.randint(0, len(replies) - 1)
    return replies[index]