# Main idea

Test if we can create two MLP that extract features from two multimodal inputs $x_1$ and $x_2$.

The first MLP will extract features from an input $x_1$. It defines a simple convolutional neural network (CNN) with two convolutional layers, max pooling, and two fully connected layers. The second MLP will extract features from an input $x_2$.  It is designed for processing text data through a simple multi-layer perceptron (MLP) architecture.  It uses an embedding layer to convert word indices into dense vectors, and then a fully connected layer with ReLU activation to process the embedded text and produce the final output. 

Finally, I define a module that combines the functionalities of the previously defined MLPs into one to process both image and text data and outputs normalized representations for both modalities (L2 normalization is applied to ensure that the representations have unit length). The goal is to learn a joint representation that captures the semantic similarity between these different modalities.

Then, I implement a loss function for this architecture. The purpose of this is to compute the contrastive loss between image and text embeddings based on cosine similarities (Related to CLIP-Model Loss Function). This ContrastiveLoss function is designed to encourage similar embeddings for positive pairs (matching image and text) and dissimilar embeddings for negative pairs. It leverages cosine similarities and applies a contrastive objective using cross-entropy loss.

Overall, we define two MLP, extract features and group similar pairs meanwhile dissimilar paris are moved away for the anchor.
How we test it? We input with one sample from one side (i.e. an image) and generates its pair (i.e. the text). So, this like a classification architecture, but here we can work with both sides.

Since it was a simple prototyping, we already defined the training and architecture design. For generating side, we need to code a decoder for each side of the model.
