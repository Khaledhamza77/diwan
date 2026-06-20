"""
Download BGE-M3 and BGE-Reranker-v2-m3 to the local HuggingFace cache.
Run once before upsert_mohasaba.py or chunker.py.
"""

from huggingface_hub import snapshot_download

MODELS = [
    "BAAI/bge-m3",
    "BAAI/bge-reranker-v2-m3",
]

for model_id in MODELS:
    print(f"Pulling {model_id} ...")
    path = snapshot_download(repo_id=model_id)
    print(f"  Cached at: {path}\n")

print("Done.")
