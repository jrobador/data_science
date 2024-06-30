# Multimodal embeddings

A multimodal embedding is a representation of data that integrates information from multiple modalities, such as text, images, audio, and video, into a unified, often lower-dimensional, vector space. This allows for the fusion and comparison of information across different types of data, enabling tasks that require understanding and leveraging multiple types of inputs simultaneously.

The goal is to map different modalities into a common embedding space where semantically similar concepts from different modalities are close to each other. For example, the concept of a "cat" represented by an image and the word "cat" would be close in this shared space.

## Basic functionality

A text embedding is a piece of text projected into a high-dimensional latent space. The position of our text in this space is a vector, a long sequence of numbers. Mathematically, an embedding space, or latent space, is defined as a manifold in which similar items are positioned closer to one another than less similar items. Sentences that are semantically similar should have similar embedded vectors and thus be closer together in the space.

One big problem is that the things can just get unwieldy at these high dimensions. Algorithms grind to a halt as the combinatorics explode, and the sparsity (most documents will have a count of 0 for most terms) is problematic for statistics and machine learning.
All we need, then, to project a bag of words related to some data into our latent space is a big matrix that defines how much each of the observed terms in our vocabulary contributes to each of our latent terms.

The most famous mathematically process are:

1. Latent semantic analysis (LSA), which uses the singular value decomposition (SVD) of the term-document matrix.
2. Latent Dirichlet allocation (LDA), which uses a statistical method called the Dirichlet process.

## Useful application of a text similarity architecture

We can frame a lot of useful tasks in terms of text similarity.

1. Search: How similar is a query to a document in your database?
2. Spam filtering: How close is an email to examples of spam?
3. Content moderation: How close is a social media message to known examples of abuse?
4. Conversational agent: Which examples of known intents are closest to the user’s message?
5. Image Captioning: Did you know that META already applies it to the images you upload to its platform?
6. Cross-Modal Retrieval: Retrieving relevant items across different modalities, such as finding images that correspond to a given text description or vice versa.

## Examples of application

### Supervised Binary Classification for Fraud Detection and Abuse Detection

#### Problem Definition

- Task: Classify emails or messages as fraud/non-fraud or abuse/non-abuse.
- Data: Text data from emails or messages.
- Labels: Binary labels indicating fraud/non-fraud or abuse/non-abuse.

#### Architecture

1. *Text Processing*: Convert text data into numerical features using techniques like TF-IDF, Bag-of-Words, or more advanced methods like **word embeddings (Word2Vec, GloVe)** or **contextual embeddings (BERT, GPT)**
2. *Feature Extraction*: For multimodal scenarios, combine text features with other relevant features (e.g., metadata, user behavior).
3. *Model*: Use a binary classifier (e.g., logistic regression, SVM, neural network) to classify the data.
4. *Loss Function*: Use Binary Cross-Entropy Loss to train the model.

#### Training and Clustering

1. *Training*: Train the model on labeled data to minimize the Binary Cross-Entropy Loss.
2. *Clustering*: During training, the model learns to create clusters of similar instances in the feature space, separating fraud/abuse cases from non-fraud/non-abuse cases.

### Recommendation System for Book Suggestions

#### Problem Definition

- Task: Recommend books based on a reader's previous selections.
- Data: Text data from book descriptions or content.
- Features: Words indicating "stress" or other thematic elements.

#### Architecture

1. Text Processing: Use a Bag-of-Words model or more sophisticated methods like TF-IDF or embeddings to convert book descriptions into numerical features.
2. Feature Engineering: Identify and quantify "stress" words or other relevant themes.
3. Similarity Measure: Calculate similarity between books based on their feature vectors.

#### Recommendation

1. Initial Selection: Given a reader's previous book selection, extract its feature vector.
2. Similarity Search: Find books with feature vectors closest to the selected book in the feature space, assuming these are most likely to be of interest to the reader.

#### Enhancing with Multimodal Embeddings

Instead of relying solely on text, incorporate other modalities such as:

- Cover Images: Use image embeddings to represent book covers.
- Audio: For audiobooks, use audio embeddings to represent the spoken content.
- User Behavior: Embed user interactions and preferences as additional features.
- Joint Embedding Space: Train a model to map different modalities into a shared embedding space where similar items (books) are close together regardless of the modality.

Example: A book's text description and its cover image would have similar embeddings if they represent the same content/theme.

## Challenges

- Alignment: Ensuring that corresponding elements from different modalities are properly aligned in the embedding space.
- Scalability: Handling the large and diverse datasets often required for training multimodal models.
- Complexity: Managing the increased complexity and computational requirements of models that process multiple types of data.
