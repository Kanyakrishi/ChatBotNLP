import os
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
wnl = WordNetLemmatizer()
from utils import clean_text_lda

# Create the LDA Model 
def read_all_files(directory):
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(clean_text_lda(text))
    return documents
  
# Function to create and save LDA model
def create_and_save_lda_model(documents, dictionary, output_path="lda_model.model"):
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=15, passes=15, random_state=42)
    for index in range(lda_model.num_topics):  # Loop through each topic
        topic_representation = lda_model.print_topic(index)  # Get the topic representation
        print(f"Topic {index}: {topic_representation}")  # Print the topic

    lda_model.save(output_path)

documents = read_all_files("webpages_crawled")
dictionary = corpora.Dictionary(documents)
dictionary.save("dog_lda_dict.gensim")
create_and_save_lda_model(documents, dictionary)