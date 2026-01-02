#!/usr/bin/env python3
"""
Pre-download and cache the LLM model.

Run this ONCE before starting the app:
    PYTHONPATH=src python scripts/download_model.py

The model will be cached in ~/.cache/huggingface/hub/
Subsequent runs will load from cache instantly.
"""

import sys
sys.path.insert(0, "src")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def main():
    print("=" * 60)
    print("HalluScope Model Downloader")
    print("=" * 60)
    print()
    print(f"This will download {MODEL_ID} (~6GB)")
    print("The model will be cached in ~/.cache/huggingface/hub/")
    print()
    print("Downloading... (this may take 5-15 minutes on first run)")
    print()
    
    from halluscope.core.llm_backend import LLMBackend
    
    # This triggers the download and caches the model
    llm = LLMBackend(model_id=MODEL_ID)
    
    print()
    print("=" * 60)
    print("✅ Model downloaded and cached successfully!")
    print("=" * 60)
    print()
    print("You can now start the app:")
    print("  Terminal 1: PYTHONPATH=src uvicorn halluscope.app.api:app --port 8000")
    print("  Terminal 2: PYTHONPATH=src streamlit run src/halluscope/app/ui.py")
    print()
    
    # Quick test generation
    print("Running a quick test...")
    result = llm.generate("Hello, how are you?", max_new_tokens=20)
    print(f"Test output: {result.text[:100]}...")
    print()
    print("✅ Model is working correctly!")


if __name__ == "__main__":
    main()
