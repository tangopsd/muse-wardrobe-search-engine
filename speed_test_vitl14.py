import time
import pandas as pd
import numpy as np
import torch
import open_clip
import faiss

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-L-14")
model = model.to(device)
model.eval()

embeddings = np.load("data/image_embeddings_vitl14.npy")
ids_df = pd.read_csv("data/embedded_ids_vitl14.csv")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

def timed_search(query_text, k=10):
    start = time.time()
    with torch.no_grad():
        tokens = tokenizer([query_text]).to(device)
        text_embedding = model.encode_text(tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.cpu().numpy().astype("float32")
    scores, indices = index.search(text_embedding, k)
    elapsed = time.time() - start
    return elapsed

queries = ["sports shoes", "casual sneakers", "ethnic kurta", "leather handbag", "summer sandals"]
times = [timed_search(q) for q in queries]

print("Individual query times (seconds):", [f"{t:.4f}" for t in times])
print(f"Average query time: {np.mean(times):.4f}s")