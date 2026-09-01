import pandas as pd
import numpy as np
import torch
import open_clip
import faiss

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device)
model.eval()

# --- Load saved embeddings and metadata ---
embeddings = np.load("data/image_embeddings.npy")
ids_df = pd.read_csv("data/embedded_ids.csv")

print("Loaded embeddings:", embeddings.shape)

# --- Build FAISS index ---
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity, since vectors are normalized
index.add(embeddings)
print("FAISS index built with", index.ntotal, "vectors")

def search(query_text, k=5):
    with torch.no_grad():
        tokens = tokenizer([query_text]).to(device)
        text_embedding = model.encode_text(tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.cpu().numpy().astype("float32")

    scores, indices = index.search(text_embedding, k)

    results = ids_df.iloc[indices[0]].copy()
    results["score"] = scores[0]
    return results[["id", "productDisplayName", "subCategory", "score"]]

# --- Test queries ---
test_queries = [
    "cozy oversized sweater",
    "sports shoes",
    "casual shoes",
    "ethnic wear",
    "black handbag",
]

for q in test_queries:
    print(f"\n=== Query: '{q}' ===")
    print(search(q, k=5).to_string(index=False))