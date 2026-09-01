import pandas as pd
import os

df = pd.read_csv("data/styles.csv", on_bad_lines="skip")

# Categories chosen for meaningful semantic overlap / confusability
target_categories = [
    "Casual Shoes",   # vs Sports Shoes — visually similar, different use case
    "Sports Shoes",
    "Kurtas",         # ethnic wear stand-in
    "Tshirts",        # western wear stand-in
    "Shirts",
    "Tops",
    "Handbags",
    "Backpacks",
    "Sandals",
    "Heels",
]

subset = df[df["articleType"].isin(target_categories)].copy()
print("Rows matching target categories:", len(subset))

# Cap per category so no single category dominates the subset
subset = subset.groupby("articleType", group_keys=False).apply(
    lambda x: x.sample(min(len(x), 600), random_state=42)
)
print("Rows after capping per category:", len(subset))

# Confirm the actual image files exist for these rows
image_dir = "data/images"
subset["image_path"] = subset["id"].astype(str) + ".jpg"
subset["exists"] = subset["image_path"].apply(lambda f: os.path.exists(os.path.join(image_dir, f)))

print("Images found on disk:", subset["exists"].sum())
print("Images missing:", (~subset["exists"]).sum())

# Keep only rows where the image actually exists
final = subset[subset["exists"]].drop(columns=["exists"])
print("Final curated subset size:", len(final))

final.to_csv("data/curated_subset.csv", index=False)
print("Saved to data/curated_subset.csv")