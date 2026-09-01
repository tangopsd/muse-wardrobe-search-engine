import pandas as pd
import numpy as np
import torch
import open_clip
import faiss

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load model ----------
print("Loading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device)
model.eval()

# ---------- Load embeddings + metadata ----------
embeddings = np.load("data/image_embeddings.npy")
embedded_ids = pd.read_csv("data/embedded_ids.csv")

# embedded_ids.csv is missing articleType — pull it back in from curated_subset.csv,
# merging on id, preserving the original embedding order
full_styles = pd.read_csv("data/styles.csv", on_bad_lines="skip")[["id", "articleType"]]
ids_df = embedded_ids.merge(full_styles, on="id", how="left")

assert len(ids_df) == embeddings.shape[0], "Mismatch between embeddings and metadata rows!"

# ---------- Build FAISS index ----------
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors")


def search(query_text, k=10):
    with torch.no_grad():
        tokens = tokenizer([query_text]).to(device)
        text_embedding = model.encode_text(tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.cpu().numpy().astype("float32")
    scores, indices = index.search(text_embedding, k)
    return ids_df.iloc[indices[0]].copy()


def precision_at_k(query_text, relevant_category, k=10):
    """
    Returns Precision@k for a single query: the fraction of the top-k
    results whose articleType matches relevant_category.
    """
    results = search(query_text, k=k)
    hits = (results["articleType"] == relevant_category).sum()
    return hits / k


# ---------- Test queries, mapped to their expected ground-truth category ----------
# (query text, expected articleType)
test_queries = [
    ("sports shoes", "Sports Shoes"),
    ("casual sneakers", "Casual Shoes"),
    ("running shoes", "Sports Shoes"),
    ("high heels", "Heels"),
    ("summer sandals", "Sandals"),
    ("ethnic kurta", "Kurtas"),
    ("traditional indian wear", "Kurtas"),
    ("graphic tshirt", "Tshirts"),
    ("formal shirt", "Shirts"),
    ("leather handbag", "Handbags"),
    ("school backpack", "Backpacks"),
]

k_values = [5, 10]

print("\n=== Precision@k results ===\n")
overall_scores = {k: [] for k in k_values}

for query_text, category in test_queries:
    row_result = f"'{query_text}' (expecting: {category})  ->  "
    for k in k_values:
        p = precision_at_k(query_text, category, k=k)
        overall_scores[k].append(p)
        row_result += f"P@{k}={p:.2f}  "
    print(row_result)

print("\n=== Averages across all test queries ===")
for k in k_values:
    avg = np.mean(overall_scores[k])
    print(f"Mean Precision@{k}: {avg:.3f}")