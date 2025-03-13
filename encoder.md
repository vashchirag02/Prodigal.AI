
# Transformer - Encoder :
Author : Siddharth Krishna Gupta Kori

Date: 4th March,2025
___
**Introduction:**

The Transformer Encoder is a key component of the Transformer architecture used in natural language processing (NLP) tasks. 

## Theory: 



### 1. Features of the Encoder

- Self-Attention Mechanism - Captures relationships between words in the input.

- Feed-Forward Network - Applies transformations to refine the representations.

- Scalable Architecture - Can be stacked to create deep models.

- Parallel Computation - Processes input sequences efficiently.

### 2. Architecture of Transformer Encoder

The Transformer Encoder consists of two main parts:

**a. Self-Attention Layer**

- Helps the model focus on important words.

- Uses Query (Q), Key (K), and Value (V) matrices to compute attention scores.

- Softmax is applied to normalize the scores.

**b. Feed-Forward Network (FFN)**

- Applies linear transformations to enhance the representations.

- Introduces non-linearity with activation functions.

___
# Scaled Dot-Product Attention

## Overview
Scaled Dot-Product Attention is a core mechanism in Transformer models, allowing the network to focus on relevant parts of the input sequence. This function computes attention scores using the dot-product of query (`q`) and key (`k`) matrices, scales them, applies an optional mask, and then weights the value (`v`) matrix accordingly.

## Formula
The scaled dot-product attention is computed as follows:

```
Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) * V
```

where:
- `Q` (Query)
- `K` (Key)
- `V` (Value)
- `d_k` is the dimension of the key vectors (scaling factor for stability)
- `mask` is an optional argument to ignore certain positions (useful in decoder layers)

---

## Implementation

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product(q, k, v, mask=None):
    """
    Computes scaled dot-product attention.
    
    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, d_k)
        k: Key tensor of shape (batch, num_heads, seq_len, d_k)
        v: Value tensor of shape (batch, num_heads, seq_len, d_v)
        mask: Optional mask tensor of shape (batch, 1, 1, seq_len)
    
    Returns:
        values: Output tensor after attention is applied.
        attention: Attention scores.
    """
    d_k = q.size()[-1]  # Dimension of key vectors
    
    # Compute raw attention scores
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    print(f"scaled.size() : {scaled.size()}")
    
    # Apply mask if provided
    if mask is not None:
        print(f"-- ADDING MASK of shape {mask.size()} --")
        scaled += mask  # Broadcasting add, assuming last dimensions match
    
    # Compute attention weights
    attention = F.softmax(scaled, dim=-1)
    
    # Compute final weighted values
    values = torch.matmul(attention, v)
    
    return values, attention
```

---

## Explanation
1. **Compute raw attention scores:**
   - Matrix multiplication of `q` (query) and `k` (key), followed by scaling by `sqrt(d_k)` for numerical stability.
   - This results in a similarity score matrix of shape `(batch, num_heads, seq_len, seq_len)`.

2. **Apply optional mask:**
   - If a mask is provided, it is added to the scaled scores.
   - This is useful in transformer decoders to prevent attending to future tokens.

3. **Compute attention weights:**
   - A softmax function is applied along the last dimension (`dim=-1`), normalizing the scores.

4. **Apply attention weights to values (`v`)**:
   - Matrix multiplication between attention scores and `v` produces the final weighted representation.

---

## Example Usage
```python
batch_size = 2
num_heads = 8
seq_length = 10
d_k = 64
d_v = 64

q = torch.rand(batch_size, num_heads, seq_length, d_k)
k = torch.rand(batch_size, num_heads, seq_length, d_k)
v = torch.rand(batch_size, num_heads, seq_length, d_v)
mask = torch.ones(batch_size, 1, 1, seq_length) * -1e9  # Example mask

values, attention = scaled_dot_product(q, k, v, mask)
print("Output Values Shape:", values.shape)
print("Attention Scores Shape:", attention.shape)
```

---

## Why Use Scaled Dot-Product Attention?
- **Captures dependencies between tokens** regardless of distance.
- **Scalability**: Efficiently handles long sequences.
- **Better numerical stability**: Scaling by `sqrt(d_k)` prevents large values from dominating softmax.

---

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al.
- PyTorch Documentation on [Softmax](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.softmax)
___

# Multi-Headed Self-Attention (MHSA)

## Overview
Multi-Headed Self-Attention (MHSA) is a fundamental component of Transformer models, allowing them to capture different types of relationships between tokens in a sequence.

## Formula
Each attention head is computed as:

```
head_i = Attention(Q_i, K_i, V_i)
```

where:
- `Q_i` (Query)
- `K_i` (Key)
- `V_i` (Value)

are derived by applying learned weight matrices to the input sequence.

The final multi-head attention output is computed as:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
```

where `W_O` is a learned projection matrix.

---

## Implementation in PyTorch
```python
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, max_sequence_length, d_model = x.size()
        print(f"x.size(): {x.size()}")
        qkv = self.qkv_layer(x)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dim)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3)
        print(f"qkv.size(): {qkv.size()}")
        q, k, v = qkv.chunk(3, dim=-1)
        print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}, ")
        values, attention = scaled_dot_product(q, k, v, mask)
        print(f"values.size(): {values.size()}, attention.size:{ attention.size()} ")
        values = values.reshape(batch_size, max_sequence_length, self.num_heads * self.head_dim)
        print(f"values.size(): {values.size()}")
        out = self.linear_layer(values)
        print(f"out.size(): {out.size()}")
        return out

```

## Explanation
1. The input sequence is projected into `Q`, `K`, and `V` matrices.
2. The `Q`, `K`, and `V` matrices are split into `num_heads` heads.
3. Attention scores are computed using scaled dot-product attention.
4. The output of each attention head is concatenated and projected.
5. The final output maintains the same shape as the input.

---

## Why Multi-Head Attention?
- **Captures diverse information**: Each head can focus on different relationships.
- **Enhances model expressiveness**: Allows the network to learn multiple attention patterns.
- **Improves parallelism**: Enables efficient processing using GPUs.

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al.
- PyTorch Documentation on [Multi-Head Attention](https://pytorch.org/docs/stable/nn.html#torch.nn.MultiheadAttention)

# Layer Normalization

## Overview
Layer Normalization (LN) is a technique used to normalize the inputs across the feature dimensions. It stabilizes training and accelerates convergence by ensuring that activations have a mean of zero and a standard deviation of one.

## Formula
Layer normalization is computed as:

```
LN(x) = gamma * ((x - mean) / sqrt(variance + eps)) + beta
```

where:
- `x` is the input tensor
- `mean` is the mean of `x` across the feature dimensions
- `variance` is the variance of `x` across the feature dimensions
- `eps` is a small constant to avoid division by zero
- `gamma` and `beta` are learnable parameters for scaling and shifting

---

## Implementation

```python
import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        """
        Implements Layer Normalization.
        
        Args:
            parameters_shape: Shape of the parameters to normalize
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))  # Scale
        self.beta = nn.Parameter(torch.zeros(parameters_shape))  # Shift

    def forward(self, inputs):
        """
        Forward pass for Layer Normalization.
        
        Args:
            inputs: Input tensor of shape (*, parameters_shape)
        
        Returns:
            Normalized output tensor of same shape as input
        """
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        print(f"Mean ({mean.size()})")
        
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        print(f"Standard Deviation ({std.size()})")
        
        y = (inputs - mean) / std
        print(f"y: {y.size()}")
        
        out = self.gamma * y + self.beta
        print(f"self.gamma: {self.gamma.size()}, self.beta: {self.beta.size()}")
        print(f"out: {out.size()}")
        
        return out
```

---

## Explanation
1. **Compute Mean and Variance:**
   - Mean is calculated across feature dimensions.
   - Variance is computed as the squared difference from the mean.

2. **Normalize the Input:**
   - Subtract the mean and divide by the standard deviation to normalize values.

3. **Apply Learnable Parameters (`gamma`, `beta`)**:
   - `gamma` (scaling factor) and `beta` (shift factor) allow the model to adaptively scale and shift the normalized values.

---

## Example Usage
```python
batch_size = 2
seq_length = 10
d_model = 512

x = torch.rand(batch_size, seq_length, d_model)
ln = LayerNormalization(parameters_shape=d_model)
output = ln(x)
print("Output Shape:", output.shape)  # Expected: (batch_size, seq_length, d_model)
```

---

## Why Use Layer Normalization?
- **Reduces Internal Covariate Shift**: Stabilizes learning by normalizing across feature dimensions.
- **Works Well for NLP Models**: Unlike Batch Normalization, Layer Normalization does not depend on batch statistics.
- **Improves Training Stability**: Helps in deep networks by keeping activations well-conditioned.

---

## References
- [Layer Normalization Paper](https://arxiv.org/abs/1607.06450) - Ba, Kiros, and Hinton
- PyTorch Documentation on [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)

# Encoder Layer in Transformers

## Overview
The **Encoder Layer** is a fundamental component of Transformer models. It consists of:
- **Multi-Head Self-Attention** for capturing dependencies between words.
- **Layer Normalization** for stabilizing training.
- **Feed Forward Network (FFN)** for transforming the representations.
- **Dropout** to prevent overfitting.
- **Residual Connections** to help with gradient flow.

This implementation follows the architecture introduced in *Attention Is All You Need* by Vaswani et al. (2017).

---

## Encoder Layer Formula
Each encoder layer follows this sequence:

```
1. Multi-Head Self-Attention → Dropout → Add & Layer Norm
2. Feed Forward Network → Dropout → Add & Layer Norm
```

Mathematically:

```
Attention Output = LayerNorm(SelfAttention(X) + X)
FFN Output = LayerNorm(FeedForward(Attention Output) + Attention Output)
```

where:
- `SelfAttention(X)` applies multi-head self-attention.
- `FeedForward(X)` is a position-wise feed-forward network.
- `LayerNorm` normalizes the output.

---

## Implementation

```python
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        """
        Implements a single Transformer Encoder Layer.
        
        Args:
            d_model: Dimensionality of the model (embedding size)
            ffn_hidden: Number of hidden units in the feed-forward network
            num_heads: Number of attention heads
            drop_prob: Dropout probability
        """
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        Forward pass through the Encoder Layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
        
        Returns:
            Output tensor of the same shape as input
        """
        residual_x = x
        print("------- ATTENTION 1 ------")
        x = self.attention(x, mask=None)
        
        print("------- DROPOUT 1 ------")
        x = self.dropout1(x)
        
        print("------- ADD AND LAYER NORMALIZATION 1 ------")
        x = self.norm1(x + residual_x)
        
        residual_x = x
        print("------- FEED FORWARD NETWORK ------")
        x = self.ffn(x)
        
        print("------- DROPOUT 2 ------")
        x = self.dropout2(x)
        
        print("------- ADD AND LAYER NORMALIZATION 2 ------")
        x = self.norm2(x + residual_x)
        
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        """
        Implements the Transformer Encoder consisting of multiple Encoder Layers.
        
        Args:
            d_model: Dimensionality of the model (embedding size)
            ffn_hidden: Number of hidden units in the feed-forward network
            num_heads: Number of attention heads
            drop_prob: Dropout probability
            num_layers: Number of encoder layers
        """
        super().__init__()
        self.layers = nn.Sequential(
            *[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)]
        )

    def forward(self, x):
        """
        Forward pass through the entire Encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
        
        Returns:
            Encoded representation
        """
        x = self.layers(x)
        return x
```

---

## Explanation
### **1. Multi-Head Self-Attention**
- Helps the model learn relationships between words.
- Each attention head computes self-attention separately and combines the outputs.
- This captures different types of dependencies in a sequence.

### **2. Layer Normalization & Dropout**
- **Normalization** ensures stable training and avoids vanishing gradients.
- **Dropout** randomly removes some connections to prevent overfitting.

### **3. Feed Forward Network (FFN)**
- Applies two linear transformations with a non-linearity (ReLU or GELU) in between.
- Processes each position separately but identically.

### **4. Residual Connections**
- Skip connections help in better gradient flow.
- Makes training deeper networks easier.

---

## Example Usage
```python
batch_size = 2
seq_length = 10
d_model = 512
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 6

x = torch.rand(batch_size, seq_length, d_model)
encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
output = encoder(x)
print("Output Shape:", output.shape)  # Expected: (batch_size, seq_length, d_model)
```

---

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al.
- PyTorch Documentation on [nn.TransformerEncoderLayer](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)