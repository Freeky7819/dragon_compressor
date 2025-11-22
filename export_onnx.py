# export_onnx.py
import torch
from dragon.interface import Dragon
import os

def export_dragon():
    print("🐲 Preparing ONNX export...")
    dragon = Dragon()
    model = dragon.model.cpu() # For export it's easier on CPU
    model.eval()

    # Dummy input (must match dimensions)
    # Batch=1, Seq=128, Dim=384
    dummy_input = torch.randn(1, 128, 384)

    output_path = "dragon_1_16.onnx"

    print(f"🛠️ Exporting to {output_path} ...")
    
    # Since you have dynamic 'ratio' (topk), ONNX export is a bit tricky.
    # We'll fix ratio to 16 for this export.
    # If you want dynamic k, you need to change the forward method for tracing.
    # For this demo we'll use fixed logic in the model or a wrapper.
    
    torch.onnx.export(
        model.compress,             # Method to export
        (dummy_input, 16),          # Arguments (Tensor, ratio int)
        output_path,
        export_params=True,
        opset_version=14,           # Newer opset for TopK support
        do_constant_folding=True,
        input_names=['input_embeddings', 'ratio'],
        output_names=['compressed_vectors', 'positions'],
        dynamic_axes={
            'input_embeddings': {0: 'batch_size', 1: 'sequence_length'},
            'compressed_vectors': {0: 'batch_size', 1: 'num_compressed'}
        }
    )
    print("✅ Successfully exported! This file can be used in C++, C#, Rust, JS...")

if __name__ == "__main__":
    export_dragon()