<div align="center">

# 🐉 Dragon Compressor

### Neural Semantic Compression for Infinite AI Context

*Resonant Pointer Architecture achieving 16:1 compression with 90%+ semantic fidelity*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Freeky7819/dragon_compressor/pulls)

</div>

---

## 📑 Table of Contents

- [What is Dragon Compressor?](#-what-is-Dragon-Compresso)
- [Key Features](#-key-features)
- [Performance Benchmarks](#-performance-benchmarks)
- [Quick Start](#-quick-start)
- [Use Cases](#-use-cases)
- [Architecture Deep Dive](#️-architecture-deep-dive)
- [Training Methodology](#-training-methodology)
- [Project Structure](#-project-structure)
- [Running Tests & Benchmarks](#-running-tests--benchmarks)
- [Docker Deployment](#-docker-deployment)
- [ONNX Export](#-onnx-export-crust-integration)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)
- [Roadmap](#-roadmap)

---

<br>

## 🎯 What is Dragon Compressor?

**Dragon Compressor** solves one of the most critical problems in modern AI: **memory management for long conversations and large document collections**.

> **💡 Key Insight:** Standard RAG systems store every sentence as a 384-dimensional vector. For 100,000 documents, that's **18.4 GB of RAM**. Dragon Compresso compresses this to **1.15 GB** while preserving 90%+ of the semantic meaning.

<br>

### The Problem
- 💾 Current RAG systems waste massive memory on redundant information
- 🐌 Vector databases slow down as they scale to millions of embeddings
- 🔥 GPU memory limits force us to choose between context window and batch size

<br>

### The Solution

> **🎯 Resonant Pointer Architecture** - instead of storing all tokens, it intelligently selects and preserves only the semantic "resonance points" that carry essential meaning.

**Think of it like this:**
- 📚 **Traditional approach**: Photocopy every page of a book
- 🎯 **Dragon Compressor**: Extract only the key insights, quotes, and turning points

<br>

---

<br>

## ✨ Key Features

### 🚀 Extreme Compression
- **16:1 ratio** (production-ready): Compress 128 tokens → 8 semantic anchors
- **64:1 ratio** (experimental): Compress 128 tokens → 2 core concepts
- Maintains **90%+ cosine similarity** to original embeddings

### 🧠 Intelligent Selection
- **Resonant Pointer Mechanism**: Multi-phase attention finds the most important information
- **Harmonic Injection**: Novel ω=6 frequency stabilization for structural coherence
- **Soft Neighbor Mixing**: Captures contextual information around key points

### ⚡ Production Ready
- **Pre-trained models** included (32MB)
- **FastAPI server** for microservice deployment
- **ONNX export** for C++/Rust/JavaScript integration
- **Full test suite** with benchmarks

### 🔬 Research Foundation
Built on rigorous mathematical principles:
- Hexagonal Base-6 harmonic logic (ω≈6.0)
- Teacher-Student knowledge distillation
- Multi-phase resonant pointer networks

<br>

---

<br>

## 📊 Performance Benchmarks

> **⚡ TL;DR:** 16:1 compression ratio, 90%+ semantic fidelity, 100 sentences/sec, 93.8% memory savings

<br>

### Memory Savings
| Documents | Standard (Float32) | Dragon 1:16 | Dragon 1:64 | Savings |
|-----------|-------------------|-------------|-------------|---------|
| 10,000    | 1.84 GB          | 0.12 GB     | 0.03 GB     | 93.5%   |
| 100,000   | 18.4 GB          | 1.15 GB     | 0.29 GB     | 93.8%   |
| 1,000,000 | 184 GB           | 11.5 GB     | 2.9 GB      | 93.8%   |

### Quality Metrics (1:16 Ratio)
- **Semantic Fidelity**: 0.91 average cosine similarity
- **Technical Content**: 0.93 (neural networks, algorithms)
- **Conversational**: 0.89 (natural dialogue)
- **Abstract Text**: 0.90 (philosophy, literature)

### Speed
- **Throughput**: ~100 sentences/second (RTX 5070)
- **Latency**: ~10ms per sentence
- **Batch Processing**: Scales linearly with GPU memory

<br>

---

<br>

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Freeky7819/dragon_compressor.git
cd dragon_compressor

# Install dependencies
pip install -r requirements.txt

# Install dragon_compressor
pip install -e .
```

<br>

### Basic Usage

```python
from dragon.interface import Dragon

# Initialize (auto-loads pre-trained Dragon Pro 1:16)
compressor = Dragon()

# Compress a sentence
text = "Artificial intelligence is transforming how we process and store information in vector databases."

result = compressor.compress(text, ratio=16)

print(f"Original: 128 tokens")
print(f"Compressed: {result['compressed_vectors'].shape[1]} semantic anchors")
print(f"Compression: 16:1 ratio")
print(f"Positions: {result['positions']}")  # Where in text are the key points?
```

**Output:**
```
Original: 128 tokens
Compressed: 8 semantic anchors
Compression: 16:1 ratio
Positions: tensor([0.0234, 0.1523, 0.3125, ...])
```

<br>

---

<br>

## 📖 Use Cases

### 1️⃣ RAG Systems - Massive Document Libraries
```python
from dragon.interface import Dragon
import numpy as np

dragon = Dragon()
documents = load_your_documents()  # List of 100k+ documents

# Compress entire corpus
compressed_db = []
for doc in documents:
    result = dragon.compress(doc, ratio=16)
    compressed_db.append({
        'vectors': result['compressed_vectors'],
        'positions': result['positions'],
        'original_id': doc.id
    })

# Save compressed database (93% smaller!)
np.save('compressed_knowledge_base.npy', compressed_db)
```

<br>

### 2️⃣ Long-Term AI Memory
```python
# Compress conversation history for AI agents
conversation_history = [
    "User: What's the weather like?",
    "AI: It's sunny and 72°F.",
    "User: Should I bring an umbrella?",
    # ... 1000+ messages
]

# Compress old messages (keeps recent ones full-resolution)
old_messages = conversation_history[:-50]
compressed_memory = dragon.compress(" ".join(old_messages), ratio=16)

# AI can still "remember" key points without storing everything
```

<br>

### 3️⃣ API Microservice
```bash
# Start the FastAPI server
python API/server.py
```

```python
# Client usage
import requests

response = requests.post('http://localhost:8000/compress', json={
    'text': 'Your document here...',
    'ratio': 16
})

compressed = response.json()
```

<br>

---

<br>

## 🏗️ Architecture Deep Dive

Dragon Compressor consists of three core components:

### 1. Harmonic Injector
Adds a **decaying sinusoidal signal** (ω=6.0) to embeddings, creating structural "landmarks" that survive compression.

```python
# Hexagonal harmonic (not circular 2π)
signal = exp(-γt) × sin(6.0t + π/3)
```

**Why ω=6?** Research shows that hexagonal frequency creates more stable interference patterns in discrete latent spaces than traditional positional encodings.

<br>

### 2. Multi-Phase Resonant Pointer
Instead of standard attention, uses a **multi-phase scanning mechanism**:
- **Phase 1**: Broad scan for high-energy semantic regions
- **Phase 2**: Refined selection with LSTM memory feedback
- **Confidence Gating**: Dynamic weighting based on information density

<br>

### 3. Soft Neighbor Mixer
When a pointer selects a "key point," it also captures surrounding context using depth-wise convolutions with dilation.

```python
# Captures ±3 token context around each selected point
Conv1D(kernel=3, padding=1) → GELU → Conv1D(kernel=3, dilation=2)
```

<br>

---

<br>

## 🔬 Training Methodology

Dragon Compressor was trained using **Teacher-Student** distillation:

| Component | Details |
|-----------|---------|
| **Teacher Model** | `all-MiniLM-L6-v2` (384-dim, Hugging Face) |
| **Dataset** | WikiText-2 (2M tokens, diverse topics) |
| **Loss Function** | Cosine Similarity + Position Regularization |
| **Optimizer** | AdamW (lr=1e-4, weight_decay=0.01) |
| **Training Time** | ~4 hours on RTX 5070 |

**Validation Protocol:**
- 80/20 train/test split
- Early stopping on validation cosine similarity
- Final model selected at epoch with best fidelity/compression tradeoff

<br>

---

<br>

## 📁 Project Structure

```
dragon_compressor/
├── dragon/
│   ├── __init__.py
│   ├── model.py              # Core architecture (Resonant Pointer)
│   ├── interface.py          # High-level API (Dragon class)
│   └── weights/
│       └── dragon_pro_1_16.pth  # Pre-trained model (32MB)
├── API/
│   └── server.py             # FastAPI microservice
├── demo.py                   # Interactive demo
├── eval_dragon_benchmark.py  # Full benchmark suite
├── export_onnx.py            # ONNX export for production
├── test_everything.py        # Unit + integration tests
├── requirements.txt
├── setup.py
├── Dockerfile                # Container deployment
└── README.md
```

<br>

---

<br>

## 🧪 Running Tests & Benchmarks

### Full Test Suite
```bash
python test_everything.py
```

**Tests include:**
- ✅ Package import verification
- ✅ Compression tensor shape validation
- ✅ API endpoint functionality
- ✅ ONNX export compatibility

<br>

### Comprehensive Benchmark
```bash
python eval_dragon_benchmark.py
```

**Benchmark output:**
- 📊 Semantic fidelity across diverse text types
- 🔍 Pointer interpretability ("X-ray" visualization)
- 💾 Memory usage calculations
- ⚡ Throughput measurements

<br>

### Interactive Demo
```bash
python demo.py
```

<br>

---

<br>

## 🐳 Docker Deployment

```bash
# Build image
docker build -t dragonmemory:latest .

# Run container
docker run -p 8000:8000 dragonmemory:latest

# Test API
curl -X POST http://localhost:8000/compress \
  -H "Content-Type: application/json" \
  -d '{"text": "Test compression", "ratio": 16}'
```

<br>

---

<br>

## 🔧 ONNX Export (C++/Rust Integration)

```python
python export_onnx.py
```

This generates `dragon_1_16.onnx` which can be loaded in:
- **C++**: ONNX Runtime
- **Rust**: tract or onnxruntime-rs
- **JavaScript**: onnxruntime-web
- **C#**: ML.NET

**Example (C++):**
```cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

Ort::Env env;
Ort::Session session(env, "dragon_1_16.onnx", Ort::SessionOptions());

// Run inference
auto output = session.Run(...);
```

<br>

---

<br>

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Areas of Interest
- 🎯 **Adaptive Compression**: Dynamic ratio selection based on content complexity
- 🌐 **Multilingual Support**: Extend beyond English (currently optimized for English)
- 📊 **Benchmark Datasets**: Test on domain-specific corpora (medical, legal, code)
- ⚡ **Performance**: CUDA kernel optimizations, quantization

<br>

### Development Setup
```bash
# Fork & clone
git clone https://github.com/Freeky7819/dragon_compressor.git

# Create feature branch
git checkout -b feature/your-feature-name

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests before committing
python test_everything.py

# Submit PR
git push origin feature/your-feature-name
```

<br>

---

<br>

## 📚 Citation

If you use DragonMemory in your research, please cite:

```bibtex
@software{dragonmemory2024,
  title={DragonMemory: Resonant Semantic Compression for Infinite AI Context},
  author={Žakelj, Damjan},
  year={2024},
  url={https://github.com/Freeky7819/dragon_compressor},
  note={Neural architecture achieving 16:1 compression with 90\%+ semantic fidelity}
}
```

**Related Research:**
- Hexagonal Harmonic Stabilization in Latent Spaces (ω=6 phenomenon)
- Multi-Phase Resonant Pointer Networks
- Teacher-Student Distillation for Semantic Compression

<br>

---

<br>

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

**TL;DR:** Free to use, modify, and distribute. Commercial use allowed. Just keep the license notice.

<br>

---

<br>

## 🙏 Acknowledgments

- **Sentence-Transformers**: For the excellent `all-MiniLM-L6-v2` model
- **PyTorch Team**: For the incredible deep learning framework
- **FastAPI**: For the elegant API framework
- **Community**: For testing, feedback, and contributions

<br>

---

<br>

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Freeky7819/dragon_compressor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Freeky7819/dragon_compressor/discussions)
- **Email**: your.email@example.com

<br>

---

<br>

## 🎯 Roadmap

### v1.1 (Q1 2025)
- [ ] Adaptive ratio selection based on content entropy
- [ ] Multi-GPU training support
- [ ] Quantized models (INT8, FP16)

<br>

### v1.2 (Q2 2025)
- [ ] Multilingual models (50+ languages)
- [ ] Online learning / incremental compression
- [ ] Integration with LangChain & LlamaIndex

<br>

### v2.0 (Q3 2025)
- [ ] Hierarchical compression (compress compressed vectors)
- [ ] Real-time streaming compression
- [ ] Hardware acceleration (TPU, Apple Silicon)

<br>

---

<br>

<div align="center">

### 🐉 Built with precision, passion, and mathematical poetry 💙

**Star ⭐ this repo if Dragon Compressor helps your project!**

<br>

[⬆ Back to Top](#-dragonmemory)

</div>
