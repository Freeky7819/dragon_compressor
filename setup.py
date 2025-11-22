from setuptools import setup, find_packages
import os

# Read README.md content for long description (if it exists),
# so your package looks professional.
long_description = "Dragon Compressor - Semantic Compression for NLP"
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="dragon-compressor",
    version="1.0.0",
    description="Resonant Latent Semantic Compression for NLP (Dragon Architecture)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Authorship - as we said: You and I (or enter your name)
    author="You and I",
    # author_email="your.email@example.com",  # Optional
    
    # Automatically finds the "dragon" package
    packages=find_packages(),
    
    # --- KEY FOR YOUR MODEL ---
    # This tells setuptools to respect the MANIFEST.in file
    # and include dragon_pro_1_16.pth in the installation.
    include_package_data=True,
    
    # Required libraries
    install_requires=[
        "torch>=2.0.0",              # We need at least PyTorch 2.0
        "transformers>=4.30.0",      # For Hugging Face models
        "sentence-transformers",     # For the teacher (all-MiniLM-L6-v2)
        "numpy"
    ],
    
    # Required Python version
    python_requires=">=3.9",
    
    # Classifiers (metadata)
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)