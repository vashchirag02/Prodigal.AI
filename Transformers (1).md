# Attention Is All You Need

**Author:** Mannu Yadav  
**Date:** February 12, 2025 

## Introduction
Transformers have revolutionized the field of deep learning, particularly in natural language processing (NLP) and computer vision. Introduced in the paper *Attention is All You Need* by Vaswani et al., 
Transformers allow significantly more parallelization. 

In the Transformer, sequential computations are reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions—an effect we counteract with Multi-Head Attention.

Let's break down the architecture of the Transformers introduced in this paper. 
A special thanks to Jay Alammar, whose blog helped me understand this paper more deeply.

---
### Key Components of a Transformer
There are mainly two components of a Transformer: Encoders and Decoders.

1. The number of Encoders proposed in the paper is 6 (based on various research studies, but we consider this a hyperparameter for future research work). 
   - An Encoder consists of two components:
     1. **Self-Attention**
     2. **Feed-Forward Neural Network**
   - Both are connected with Layer Normalization.

2. The number of Decoders proposed in this paper is also 6, the same as the Encoders. 
   - There are three major sections in Decoders:
     1. **Self-Attention**
     2. **Encoder-Decoder Self-Attention**
     3. **Feed-Forward Neural Network**
   - After the last Decoder, there is a Linear and Softmax Layer at the end.

This is the basic breakdown of the architecture. Now, let's understand the working of each layer.
Below is an image of the architecture.

---
## Image Representation of a Transformer
Below is a visual representation of a Transformer model:

![Transformer Model](https://github.com/MaNNu-yadav/Prodigal_Intersnhip_Task/blob/main/Transformers/images/Screenshot%202025-02-12%20193257.png?raw=true)
![Expanded Architecture](https://github.com/MaNNu-yadav/Prodigal_Intersnhip_Task/blob/main/Transformers/images/Screenshot%202025-02-12%20193356.png?raw=true)


---
## Positional Encoding
Since our model contains no recurrence and no convolution, in order for the model to make use of the
order of the sequence, we must inject some information about the relative or absolute position of the
tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension **d_model** as the embeddings so that the two can be summed.

In this paper, researchers proposed sine and cosine functions of different frequencies for position encoding and also used learned positional embeddings algorithms—both produced similar results.

## Converting Input Words into Vectors
We begin by turning each input word into a vector using an embedding algorithm.

We begin by converting each input word into a vector using an **embedding algorithm**.  

- The embedding process occurs only in the **bottom-most encoder**.  
- Each encoder receives a list of vectors, each of **size 512**:  
  - In the **bottom encoder**, these vectors represent **word embeddings**.  
  - In other encoders, they represent the **output of the encoder directly below**.  
- The size of this list is a **hyperparameter** that can be set, typically matching the **length of the longest sentence** in the training dataset.  

After embedding the words in the input sequence, each word flows through the **two layers** of the encoder.


After embedding the words in our input sequence, each of them flows through each of the two layers of the encoder.

## Encoder
The encoder is composed of a stack of **N = 6** identical layers. Each layer has two sub-layers:

1. **Multi-Head Self-Attention Mechanism**
2. **Fully Connected Feed-Forward Network**

A residual connection is applied around each of the two sub-layers, followed by Layer Normalization.

That is, the output of each sub-layer is:
$$
\text{LayerNorm}(x + \text{Sublayer}(x))
$$

## Self-Attention
The first step in calculating self-attention is to create three vectors from each of the encoder’s input vectors. So, for each word, we create a **Query vector (Q)**, a **Key vector (K)**, and a **Value vector (V)**. These vectors are created by multiplying the embedding by three matrices that we train during the training process. These matrices are initialized randomly at first and updated during training.

These new vectors are smaller in dimension than the embedding vector. Their **dimensionality is 64**, while the embedding and encoder input/output vectors have a dimensionality of 512.

After calculating the **Query (Q), Key (K), and Value (V) vectors**, the second step is to calculate a score.
The score is calculated by taking the dot product of the query vector with the key vector of the respective word we’re scoring.

After that, we divide the score by $\sqrt{d_k}$ and apply a softmax function to obtain the weights on the values.

This step is important because for large values of **d_k**, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $\sqrt{d_k}$.

The resulting vector (**Z**) is then sent to the feed-forward neural network.

$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

## Why Self-Attention?
1. It captures long-range dependencies—Self-attention allows every word to focus on every other word, no matter how far apart they are.
2. Self-attention processes all words at the same time, making it much faster and more efficient.
3. Self-attention helps the model understand the meaning based on surrounding words.

---

## Multi-Head Attention

The paper further refines the self-attention layer by introducing a mechanism called **multi-headed attention**. This enhances the performance of the attention layer in two ways:  

1. It expands the model’s ability to focus on different positions.  
2. It provides the attention layer with multiple **representation subspaces**.  

**Multi-Headed Attention** applies the same self-attention calculation **eight times**, each with different weight matrices.  

This results in **eight Z matrices**, but the feed-forward layer expects a **single matrix**, not multiple ones. To resolve this, we:  

- **Concatenate** the eight matrices.  
- Multiply the concatenated output by an additional weight matrix **W_O** to produce the final output.  


Instead of performing a single self-attention operation, multi-head attention applies eight parallel attention mechanisms. The outputs are concatenated and transformed using an additional weight matrix \( W_O \).

---

## Feed-Forward Network (FFN)

Each position in the Transformer model has its own **Feed-Forward Network (FFN)**, which is **applied independently** to every position in the sequence.  

This network consists of **two linear transformations** with a **ReLU activation** function in between:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2
$$

Although the **same transformation** is applied across all positions in the sequence, the parameters vary from **layer to layer**.


---

## Residual Connections

Each sub-layer (self-attention, FFN) in every encoder has a residual connection followed by layer normalization:

$$
\text{LayerNorm}(X + Z)
$$

---

## Decoder

- The decoder is composed of a stack of **N = 6** identical layers.  
- In addition to the two sub-layers in each encoder layer, the decoder inserts a **third sub-layer**, which performs **multi-head attention** over the output of the encoder stack.  

- The **encoder** processes the input sequence:  
  - The output of the top encoder is transformed into a set of **attention vectors**: **K** (Keys) and **V** (Values).  
  - These vectors are used by each decoder in the **encoder-decoder attention** layer, helping the decoder focus on appropriate parts of the input sequence.  

- Similar to the encoder, the decoder includes:  
  - **Residual connections** around each sub-layer.  
  - **Layer normalization** applied after each sub-layer.  

- The **Transformer decoder** uses a **modified self-attention mechanism** to prevent each position from attending to future positions:  
  - This is crucial for tasks like **text generation**, ensuring predictions are based only on known information.  
  - A **mask** is applied to prevent the model from accessing future tokens.  
  - When predicting the word at position **i**, the model only considers **earlier positions** and does not cheat by looking ahead.  
  - Additionally, the **output embeddings are offset by one position**, meaning:  
    - During training, the model learns to predict the **next word** using only past and present words.  
    - This ensures a natural, **step-by-step text generation process**.  

- The **Encoder-Decoder Attention** layer:  
  - Functions like **multi-headed self-attention**.  
  - **Queries (Q) matrix** is created from the layer below it.  
  - **Keys (K) and Values (V) matrices** are taken from the **encoder stack output**.  


---

## Linear and Softmax Layers

The decoder outputs a vector of float values, which is then processed by:
1. **Linear Layer**:    The Linear layer is a simple fully connected neural network that projects the vector produced by the stack of decoders, into a much, much larger vector called a logits vector.
2. **Softmax Layer**: Converts logits into probabilities, where the highest-probability word is selected as output.

---

## Optimizer

The paper uses the **Adam Optimizer**,
Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
with the following learning rate schedule:

$$
\text{Learning Rate} = d_{\text{model}} \cdot \min \left( \mathrm{step\_num}^{-0.5}, \mathrm{step\_num} \cdot \mathrm{warmup\_steps}^{-1.5} \right)
$$

- **Warmup steps** = 4000
- **Beta1** = 0.9
- **Beta2** = 0.98
- **Epsilon** = $({10}^{-9})$

---

## Results

1. On the **WMT 2014 English-to-German** translation task, the Transformer outperformed previous models (ByteNet, ConvS2S) by **2.0+ BLEU**, achieving **28.4 BLEU**.
2. On the **WMT 2014 English-to-French** translation task, the Transformer achieved **41.0 BLEU**, surpassing all previous models at **1/4th the training cost**.

---

## Conclusion

This work presented the **Transformer**, the first sequence transduction model based entirely on  
attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with  
multi-headed self-attention.  

The **Transformer** can be trained significantly faster than architectures based on recurrent or convolutional layers.  
On both **WMT 2014 English-to-German** and **WMT 2014 English-to-French** translation tasks, it achieve a new state-of-the-art.  
Transformers are the future of AI; their unique functioning will benefit various domains like **computer vision**, **natural language processing (NLP)**, and **reinforcement learning**.  
With their ability to **model long-range dependencies** and process large amounts of data efficiently, Transformers have revolutionized tasks such as **image recognition**, **text generation**, **speech processing**.  
By eliminating the need for recurrence, they enable parallel processing, making training significantly faster and more scalable across different industries, from healthcare to finance and beyond.


