# README.md

## Introduction
Developed an NLP-based chatbot focused on DOGS, capable of engaging in domain-specific conversations and extracting information from the internet or a structured knowledge base.

## System Description
Designed for user-friendly interactions, this chatbot employs advanced NLP methodologies to deliver relevant responses, enhancing user experience through real-time web searches and personalized suggestions for unresolved queries.

## NLP Techniques

### 1. Part-of-Speech Tagging
   - **Overview**: Isolates nouns from queries and knowledge base entries for analysis.
   - **Application**: 
     - Extracts nouns for further analysis.
     - Determines similarity by noun overlap, ensuring relevance.
     - Ranks responses based on noun overlap and user preferences.

### 2. Cosine Similarity
   - **Overview**: Assesses text similarity to match user queries with database facts.
   - **Application**: 
     - Calculates similarity using TF-IDF vectors.
     - Applies a relevance threshold to ensure fact pertinence.
     - Prioritizes facts by similarity score, aligning with user preferences.

### 3. Dependency Parsing
   - **Overview**: Analyzes sentence structure for grammatical relationships.
   - **Application**: 
     - Extracts entities and their relationships.
     - Matches queries with facts by evaluating overlaps.
     - Ranks facts by relevance to the user's context.

### 4. LDA Topic Modeling
   - **Overview**: Groups similar words into topics from large text volumes.
   - **Application**: 
     - Identifies significant text topics.
     - Matches user queries with knowledge base topics based on relevance.
     - Selects the most pertinent topic and facts for responses.

### 5. NER Algorithm
   - **Overview**: Enhances text understanding by categorizing key elements.
   - **Application**: 
     - Isolates named entities from queries and knowledge base.
     - Identifies overlaps for semantic matching.
     - Prioritizes candidate facts for response determination.
