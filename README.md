# 🐉 Dragon Compressor
### Resonant Semantic Compression for Infinite Context

> "Storing the entire history is wasteful. Storing the resonance of memory is an art."

**Dragon Compressor** is a specialized neural architecture for extreme compression of semantic vectors (embeddings). Instead of simple averaging or *pooling*, Dragon uses a **Resonant Pointer** mechanism that finds "needles in a haystack" within the latent space and compresses them into a compact form.

The result? **90% less memory usage** in RAG systems and long-term memory of AI agents, while preserving key information.

---

## 🚀 Key Features

* **Extreme Compression:** Supports ratios from **1:16** to **1:64**. Turns 128 tokens into 8 or even 2 vectors.
* **Harmonic Injection:** Uses sine-exponential signal injection directly into the *hidden state*, enabling the model to better understand position and rhythm in text.
* **Multi-Phase Pointer:** Multi-phase decision mechanism that iteratively selects the most important parts of information.
* **Plug & Play:** Comes with a pre-trained **Dragon Pro 1:16** model (32MB), ready for immediate use.

## 📦 Installation

Install the package directly from the directory:

```bash
pip install .
```

Note: Installation automatically includes optimized weights (32MB).

## ⚡ Quick Start

Usage is simple. Dragon works as a wrapper around your NLP model.

```python
from dragon.interface import Dragon

# 1. Initialization (automatically loads weights)
compressor = Dragon()

text = "Artificial intelligence transforms the world, but long contexts remain a problem."

# 2. Compression (Ratio 1:16 - default)
# This transforms a long sentence into a few key vectors.
result = compressor.compress(text, ratio=16)

vectors = result['compressed_vectors']  # Tensor of shape [1, k, 384]
positions = result['positions']         # Where in the text were these informations?

print(f"Compressed to {vectors.shape[1]} vectors.")
```

## 🧠 Architecture

Dragon is not a regular autoencoder. It consists of three unique modules:

**Harmonic Injector:** Adds a "heartbeat" (decaying sinusoidal signal) to input data, helping to preserve sequence at high compression.

**Resonant Pointer (Multi-Phase):** Instead of Self-Attention on all tokens, the pointer "scans" the text and selects only points with high resonance (importance).

**Soft Neighbor Mixer:** When the pointer selects a point, it also "picks up" information from nearby neighbors (context) before writing to memory.

## 📊 Performance

The Dragon Pro 1:16 model was trained on the WikiText-2 dataset using the "Teacher-Student" method (teacher: all-MiniLM-L6-v2).

* **Teacher:** all-MiniLM-L6-v2 (384 dim)
* **Architecture:** Dragon v3.7 (Resonant)
* **Model Size:** 32 MB

## 🔬 The Science: Why w=6?

Dragon uses a novel **Harmonic Injection** mechanism based on Hexagonal Base-6 Logic rather than standard circular ($2\pi$) trigonometry. 

Our research indicates that in discrete latent spaces, setting the harmonic frequency to **$\omega=6$** creates structurally stable embeddings ("crystallization") that survive high compression ratios better than standard positional encodings.

**Benchmark Results (Ratio 1:16):**
- **Cosine Fidelity:** 0.90+ (vs 0.78 with standard pooling)
- **Throughput:** ~100 sentences/sec (RTX 5070)
- **Memory:** 94% reduction vs float32

## 🛠️ Development

This project was created as part of research on the evolution of memory in AI systems. The goal is to enable agents to "sleep" – processing and compressing daily interactions into long-term memory that doesn't occupy terabytes of space.