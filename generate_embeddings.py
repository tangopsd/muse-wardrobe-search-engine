import pandas as pd
import numpy as np
import torch
import open_clip
from PIL import Image
import os
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Loading CLIP ViT-B/14...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)
model = model.to(device)
model.eval()
print("Model loaded.")

#load subset
df = pd.read_csv("data/curated_subset.csv")
print(f"Encoding {len(df)} images..")

image_dir = "data/images"
embeddings = []
valid_ids = []

start = time.time()

with torch.no_grad():
    for i, row in df.iterrows():
        img_id = row["id"]
        img_path = os.path.join(image_dir, f"{img_id}.jpg")

        try:
            image = Image.open(img_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            embedding = model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # normalize
            embeddings.append(embedding.cpu().numpy().flatten())
            valid_ids.append(img_id)
        except Exception as e:
            print(f"Skipping {img_id}: {e}")

        if len(valid_ids) % 500 == 0 and len(valid_ids) > 0:
            elapsed = time.time() - start
            print(f"  {len(valid_ids)} done, {elapsed:.1f}s elapsed")

print(f"Finished encoding {len(valid_ids)} images in {time.time() - start:.1f}s")

# results - save
embeddings = np.array(embeddings, dtype="float32")
np.save("data/image_embeddings_vitl14.npy", embeddings)

ids_df = df[df["id"].isin(valid_ids)].copy()
ids_df = ids_df.set_index("id").loc[valid_ids].reset_index()
ids_df.to_csv("data/embedded_ids_vitl14.csv", index=False)

print("Saved embeddings shape:", embeddings.shape)
print("Saved data/image_embeddings.npy and data/embedded_ids.csv")