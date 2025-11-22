import torch
from dragon.interface import Dragon
import time

def show_power():
    print("\n🐉 STARTING DRAGON COMPRESSOR DEMO...")
    
    # 1. Initialization
    # Will automatically find your dragon_pro_1_16.pth
    dragon = Dragon()
    
    # 2. Test data (Long and complex text)
    long_text = """
    The architecture of neural networks has evolved significantly over the last decade. 
    Starting with simple feed-forward networks, we moved to recurrent structures like LSTMs 
    to handle sequential data. However, the introduction of the Transformer architecture 
    marked a revolutionary leap, allowing for parallel processing of sequences via self-attention mechanisms.
    Now, with the Dragon Architecture, we are taking the next step: moving from raw attention 
    to resonant latent pointers, enabling massive compression of semantic meaning without losing the core essence.
    This allows agents to maintain infinite context windows by storing only the resonant harmonics of the past.
    """
    
    # Clean text for better output
    long_text = " ".join(long_text.split())
    print(f"\n📄 ORIGINAL TEXT ({len(long_text.split())} words):")
    print(f"'{long_text[:100]}...'")

    # 3. Live compression
    print("\n⚡ EXECUTING COMPRESSION...")
    start = time.time()
    
    # Dynamic compression 1:16 (your native mode) and extreme 1:64
    res_16 = dragon.compress(long_text, ratio=16)
    res_64 = dragon.compress(long_text, ratio=64)
    
    dt = (time.time() - start) * 1000
    
    # 4. Results
    orig_emb = dragon.nlp.encode(long_text)
    
    print(f"⏱️  Processing time: {dt:.2f} ms")
    print("-" * 50)
    print(f"Original (MiniLM): {orig_emb.shape[0]} dimensions (dense vector)")
    print(f"Dragon (1:16):     {res_16['compressed_vectors'].shape[1]} vectors (Preserved key thoughts)")
    print(f"Dragon (1:64):     {res_64['compressed_vectors'].shape[1]} vectors (Only absolute essence)")
    print("-" * 50)
    print("✅ Dragon System: READY.")

if __name__ == "__main__":
    show_power()