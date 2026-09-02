import pandas as pd
import os

df = pd.read_csv("data/styles.csv", on_bad_lines="skip")

#chosen for semantic overlap - easy to confuse
target_categories = [
    "Casual Shoes",   
    "Sports Shoes",
    "Kurtas",         
    "Tshirts",        
    "Shirts",
    "Tops",
    "Handbags",
    "Backpacks",
    "Sandals",
    "Heels",
]

subset = df[df["articleType"].isin(target_categories)].copy()
print("rows matching target categories:", len(subset))

# setting cap per category to prevent single category domination
subset = subset.groupby("articleType", group_keys=False).apply(
    lambda x: x.sample(min(len(x), 600), random_state=42)
)
print("rows after capping per category:", len(subset))

image_dir = "data/images"
subset["image_path"] = subset["id"].astype(str) + ".jpg"
subset["exists"] = subset["image_path"].apply(lambda f: os.path.exists(os.path.join(image_dir, f)))

print("images found on disk:", subset["exists"].sum())
print("images missing:", (~subset["exists"]).sum())

final = subset[subset["exists"]].drop(columns=["exists"])
print("final curated subset size:", len(final))

final.to_csv("data/curated_subset.csv", index=False)
print("saved to data/curated_subset.csv")