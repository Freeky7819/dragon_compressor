import torch
import os
from sentence_transformers import SentenceTransformer
from .model import DragonArchitecture

class Dragon:
    def __init__(self, model_path=None, device=None):
        # 1. Set device (GPU/CPU)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🐲 Dragon awakening on: {self.device}")
        
        # 2. NLP Encoder (Teacher)
        # Using all-MiniLM-L6-v2, which is compatible with the training
        self.nlp = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # 3. Dragon Architecture
        # Initialize empty architecture
        self.model = DragonArchitecture(d_model=384, max_seq_len=128).to(self.device)
        
        # 4. Weight loading logic (Smart Load)
        if model_path is None:
            # If path is not provided, find 'dragon_pro_1_16.pth' in 'weights' directory within the package
            current_dir = os.path.dirname(__file__)
            default_path = os.path.join(current_dir, 'weights', 'dragon_pro_1_16.pth')
            
            if os.path.exists(default_path):
                model_path = default_path
            else:
                raise FileNotFoundError(
                    f"❌ CRITICAL ERROR: Cannot find default model at {default_path}.\n"
                    "Did you move the 'dragon_pro_1_16.pth' file to the dragon/weights/ directory?"
                )

        print(f"📂 Loading Dragon Pro (1:16) from: {model_path}")
        
        # Safe loading (map_location ensures it works on CPU even if trained on GPU)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval() # Lock into inference mode

    def compress(self, text_or_list, ratio=16):
        """
        Main function.
        Ratio: Default 16 (since the model is 'pro_1_16'), but
        you can change it to 8 or 64 for dynamic compression.
        """
        # A. Encode with Teacher model
        token_emb = self.nlp.encode(text_or_list, output_value='token_embeddings', convert_to_tensor=True)
        
        # B. Batch preparation (Padding/Stacking)
        batch_tensors = []
        if isinstance(token_emb, list):
            batch_tensors = token_emb
        else:
            batch_tensors = [token_emb]
            
        target_len = 128
        padded = torch.zeros(len(batch_tensors), target_len, 384).to(self.device)
        
        for i, t in enumerate(batch_tensors):
            l = min(t.shape[0], target_len)
            padded[i, :l, :] = t[:l, :]
            
        # C. Dragon Compression
        with torch.no_grad():
            compressed, positions = self.model.compress(padded, ratio=ratio)
            
        return {
            "compressed_vectors": compressed.cpu(),
            "positions": positions.cpu(),
            "ratio": ratio
        }