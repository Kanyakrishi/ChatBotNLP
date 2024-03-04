from collections import deque
import os
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from utils import  preprocess_text

wnl = WordNetLemmatizer()


def crawl_and_save(start_urls, max_urls=25):
    if not os.path.exists('webpages_crawled'):
        os.makedirs('webpages_crawled')
    
    crawled_urls = set()  # crawled URLs to avoid repetition
    urls_queue = deque(start_urls)  # Queue for BFS
    
    while urls_queue and len(crawled_urls) < max_urls:
        current_url = urls_queue.popleft()  # Pop from the queue
        if current_url in crawled_urls:
            continue  # Skip ifalready been crawled
        
        try:
            response = requests.get(current_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Save the webpage content
                safe_filename = f"{current_url.split('//')[-1].replace('/', '_').replace('?', '_')}.txt"
                filepath = os.path.join('webpages_crawled', safe_filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(soup.get_text())
                
                crawled_urls.add(current_url)  # Mark this URL as crawled
                
                # Find and enqueue new URLs
                for link in soup.find_all('a', href=True):
                    if len(crawled_urls) >= max_urls:
                        break  # max urls
                    next_page = link['href']
                    if not next_page.startswith('http'):
                        next_page = urljoin(current_url, next_page)  
                    if next_page not in crawled_urls and next_page not in urls_queue:
                        urls_queue.append(next_page)  # Enqueue new URL
            elif response.status_code in [403, 404]:
                print(f"Access denied or not found for {current_url}")
            else:
                print(f"Failed to crawl {current_url} with status code: {response.status_code}")
            
        except Exception as e:
            print(f"Failed to crawl {current_url}: {e}")
        
    # print("Crawled all URLS: ", crawled_urls)
                
            
def cleaning(data_dir):
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding='utf-8') as f:
            text = f.read()
            cleaned_text = preprocess_text(text)
            # Writing back 
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)
        
def extract_top_terms(data_dir,top_n=50):
    all_content = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding='utf-8') as f:
            text = f.read()
            all_content.append(text)
    
    # Calculate TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)  
    X = vectorizer.fit_transform(all_content)
    
    # Sum TF-IDF scores for each term across all documents
    scores = np.sum(X, axis=0)
    scores = np.squeeze(np.asarray(scores))  # Convert from matrix to array
    
    # Get terms and scores
    feature_names = np.array(vectorizer.get_feature_names_out())
    sorted_indices = np.argsort(scores)[::-1]  # Indices of terms in descending order of score
    
    # Extract the top N terms with the highest TF-IDF scores
    top_terms = feature_names[sorted_indices][:top_n]
    # print(feature_names)
    return top_terms
        
        
start_urls = ["https://www.purina.co.uk/articles/dogs/behaviour/common-questions/amazing-dog-facts"]

# Part 1a & 1b Crawling URL, upto 25 urls. Saving the content to a file
crawl_and_save(start_urls)

# Part 1c Cleaning texts using NLP techniques 
cleaning('webpages_crawled')

# Part 1d  function to extract at least 25 important terms f
important_terms = extract_top_terms('webpages_crawled')
print("\nTOP TERMS: ", important_terms)

# 1e knowledge base information
dog_knowledge_base = {
    "breed": [
        "Dog breeds are distinctly varied and were originally bred for specific roles such as hunting, guarding, or companionship.",
        "The American Kennel Club recognizes over 190 dog breeds."
    ],
    "purina": [
        "Purina is a pet food brand that offers a wide range of products for dogs, including dry kibble, wet food, and treats.",
        "Purina conducts extensive pet nutrition research and operates the Purina PetCare Center for diet and feeding studies."
    ],
    "article": [
        "Articles about dogs can cover a range of topics including health, nutrition, training, and breed information.",
        "Reading articles from reputable sources can help dog owners make informed decisions about their pet's care."
    ],
    "name": [
        "Choosing a name for a dog can be based on its personality, appearance, or breed traits.",
        "Popular dog names include Max, Bella, Charlie, and Luna."
    ],
    "advice": [
        "Professional advice from veterinarians and dog trainers can be crucial for addressing health and behavior issues.",
        "Seeking advice from credible sources is important for the well-being of dogs."
    ],
    "product": [
        "Dog products range from food and treats to toys, beds, and grooming supplies.",
        "Choosing the right products for a dog's size, age, and health can improve their quality of life."
    ],
    "brand": [
        "There are numerous brands in the pet industry, each offering different types of products and food for dogs.",
        "Well-known dog food brands include Royal Canin, Purina, and Hill's Science Diet."
    ],
    "guide": [
        "Guides can provide step-by-step instructions on dog care, training, and nutrition.",
        "Puppy guides are especially helpful for first-time dog owners."
    ],
    "newsletter": [
        "Many dog-related websites and organizations offer newsletters that provide updates, advice, and stories about dogs.",
        "Subscribing to a dog-related newsletter can be a great way to stay informed about new research and tips for dog care."
    ],
    "senior": [
        "Senior dogs often require different care compared to younger dogs, including special diets and more frequent health check-ups.",
        "Age-related changes in senior dogs can include reduced mobility, hearing loss, and vision impairment."
    ],
    "puppy": [
        "Puppies require a lot of time, patience, and training during their first few months.",
        "Proper nutrition, socialization, and veterinary care are crucial for a puppy's development."
    ],
    "feeding": [
        "Feeding practices for dogs vary based on their age, breed, and health status.",
        "It's important to measure a dog's food and follow feeding guidelines to prevent obesity."
    ],
    "finding": [
        "Finding the right dog involves considering lifestyle, housing, and family members' needs and preferences.",
        "Adoption from shelters or rescues is a responsible way to find a new dog while providing a home to a pet in need."
    ],
    "owner": [
        "Dog owners are responsible for their pet’s health, safety, and well-being.",
        "Being a responsible owner includes providing regular veterinary care, proper nutrition, and adequate exercise."
    ],
    "adult": [
        "Adult dogs typically require less intensive care than puppies but still need regular exercise, health check-ups, and balanced nutrition.",
        "The transition from puppy to adult food should be gradual to avoid digestive issues."
    ],
    "topic": [
        "Topics related to dogs can range from training methods to health issues and breed-specific information.",
        "Engaging with various topics can help owners provide a better life for their dogs."
    ],
    "kitten": [
        "While primarily associated with cats, raising kittens and puppies together can lead to harmonious relationships if properly introduced.",
        "Kittens and puppies can learn to coexist peacefully with proper supervision and socialization."
    ],
    "treat": [
        "Dog treats can be used as a part of training to reinforce positive behavior.",
        "It's important to choose treats that are suitable for the dog’s size and dietary needs."
    ],
    "black": [
        "Black dogs are as diverse in breed and personality as dogs of any other color.",
        "Black Dog Syndrome is a phenomenon where black dogs are often the last to be adopted from shelters."
    ],
    "contact": [
        "In case of emergency, dog owners should have contact information for their veterinarian readily available.",
        "Microchipping dogs and having a collar with contact information can help if the dog gets lost."
    ],
    "type": [
        "Dogs come in various types including companion dogs, working dogs, and sporting dogs."]
}

# 1f Pickle the knowledge base for future use
with open("dog_facts.pickle", "wb") as f:
    pickle.dump(dog_knowledge_base, f)


