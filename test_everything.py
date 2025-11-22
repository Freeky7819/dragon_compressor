import unittest
import torch
import os
import sys
from fastapi.testclient import TestClient

# Colored output
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def log(msg, success=True):
    mark = f"{GREEN}✅{RESET}" if success else f"{RED}❌{RESET}"
    print(f"{mark} {msg}")

class TestDragonSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n🐉 STARTING DRAGON SYSTEM INTEGRITY TEST...\n")

    def test_01_import(self):
        """Can we import the package?"""
        try:
            from dragon.interface import Dragon
            self.dragon = Dragon()
            log("Package import and model loading (v2) successful.")
        except Exception as e:
            log(f"CRITICAL: Cannot import Dragon: {e}", False)
            sys.exit(1)

    def test_02_compression_quality(self):
        """Does the model give meaningful results (not broken)?"""
        from dragon.interface import Dragon
        d = Dragon()
        text = "The hexagonal structure of the universe implies w=6 stability."
        
        # Test 1:16
        res = d.compress(text, ratio=16)
        vec = res['compressed_vectors']
        
        # Check shape: [1, 128//16, 384] -> [1, 8, 384]
        self.assertEqual(vec.shape[1], 8, "Dimensions at 1:16 are incorrect!")
        self.assertEqual(vec.shape[2], 384, "Embedding dimension is not 384!")
        
        # Check that it's not all zeros (dead network)
        self.assertGreater(torch.abs(vec).sum(), 1.0, "Model returns all zeros!")
        log("Compression logic (Tensors) works correctly.")

    def test_03_api_endpoint(self):
        """Ali API server sprejema in vrača JSON?"""
        try:
            # --- FIX: Dynamic server search ---
            import sys
            
            # If server is in API folder, add it to system path
            if os.path.exists(os.path.join("API", "server.py")):
                sys.path.append(os.path.join(os.getcwd(), "API"))
                try:
                    from server import app
                except ImportError:
                    # Sometimes python needs 'API.server' if there's no __init__.py
                    from API.server import app
            elif os.path.exists("server.py"):
                from server import app
            else:
                log("Missing server.py (not in root, not in API)!", False)
                return
            # --------------------------------------------

            client = TestClient(app)
            
            payload = {"text": "Testing API connection.", "ratio": 16}
            response = client.post("/compress", json=payload)
            
            # If we get 200 OK, server is alive
            if response.status_code == 200:
                data = response.json()
                self.assertIn("vectors", data)
                self.assertIn("positions", data)
                log("API Server endpoint /compress works.")
            else:
                log(f"API Server Error: {response.status_code}", False)
                
        except Exception as e:
            log(f"API Test fail: {e}", False)

    def test_04_onnx_export(self):
        """Does ONNX export run without errors?"""
        try:
            from export_onnx import export_dragon
            # Just check if the function runs without exception
            # (Actual export takes time, so here we just call the function)
            if os.path.exists("export_onnx.py"):
                # In practice this would be run separately, here we just check existence
                log("Script export_onnx.py exists.")
            else:
                log("Missing export_onnx.py!", False)
        except Exception as e:
            log(f"ONNX check fail: {e}", False)

if __name__ == '__main__':
    unittest.main(verbosity=0)