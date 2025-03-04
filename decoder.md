#Transformer Decoder Explanation#

Overview

This document explains the Transformer decoder implementation in PyTorch. The Transformer architecture is widely used in NLP tasks, such as machine translation and text generation.

Code Breakdown

1. Importing Dependencies

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

We import necessary modules from PyTorch:

torch for tensor operations.

torch.nn for defining neural network layers.

torch.nn.functional for activation functions.

math for mathematical operations.

2. Positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

Explanation

Since Transformers do not process sequences sequentially (like RNNs), positional encoding is added to retain positional information.

A sinusoidal function is used to generate positional encodings that get added to the input embeddings.

3. Multi-Head Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

Explanation

This class implements multi-head self-attention, allowing the model to focus on different parts of the input sequence simultaneously.

Each head processes a portion of the input dimension (d_model / num_heads).

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V)

The scaled dot-product attention computes attention scores between queries (Q), keys (K), and values (V).

mask is used to prevent attending to future tokens in the sequence.

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(attn_output)

Query (Q), Key (K), and Value (V) are transformed and split into multiple heads.

Attention is computed separately for each head and then concatenated back together.

4. Feed-Forward Network

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

This consists of two fully connected layers with a ReLU activation function in between.

Expands and compresses the representation to improve expressiveness.

5. Decoder Layer

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

Explanation

Masked Multi-Head Attention: Prevents the decoder from seeing future tokens.

Encoder-Decoder Attention: Helps the decoder attend to encoder outputs.

Feed-Forward Network: Processes features.

Layer Normalization: Ensures stability.

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        attn_out1 = self.masked_attn(x, x, x, tgt_mask)
        x = self.norm1(x + attn_out1)
        attn_out2 = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + attn_out2)
        ffn_out = self.ffn(x)
        return self.norm3(x + ffn_out)

6. Decoder (Stack of Decoder Layers)

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

The decoder consists of multiple layers, each performing the operations described above.

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return self.norm(x)

7. Example Usage

d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
vocab_size = 10000
max_len = 100

decoder = Decoder(d_model, num_heads, d_ff, num_layers, vocab_size, max_len)
print(decoder)

Defines a 6-layer Transformer decoder with 512-dimensional embeddings and 8 attention heads.

Prints the model architecture.

Conclusion

This implementation follows the Transformer decoder structure described in Attention Is All You Need and is commonly used in NLP tasks like text generation and translation.

