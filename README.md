# Muse: Text-to-Image Fashion Search Engine

## Overview

Muse is a text-to-image wardrobe and accessory search engine: type a phrase like "cozy oversized sweater" or "black leather handbag," and get back the most visually and semantically matching product images from a curated dataset. It uses a pretrained CLIP model to convert both text and images into comparable numeric vectors, then a FAISS index to find the closest matches in real time. Unlike a keyword search, Muse understands meaning: it can match "gym shoes" to sports shoes even though the word "athletic" or "sports" never appears in the product name.

## How CLIP Works

CLIP learns to place images and text in the same vector space using two encoders: a ResNet or Vision Transformer (this project uses ViT-B/32 and ViT-L/14) for images, and a Transformer for text. Both are trained together via contrastive learning: shown batches of image-caption pairs, the model learns to score true pairs as highly similar and mismatched pairs as dissimilar, using cosine similarity, a measure of the angle between two vectors, not their distance.

This project uses CLIP's pretrained encoders as-is (no training required): each image is encoded once and stored, and a text query is encoded at search time into that same space. FAISS then finds the closest image vectors by cosine similarity, which is what powers the actual search.

## Dataset

Muse is built on the [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset from Kaggle, which contains 44,000+ product images with structured metadata (category, gender, article type, season, etc.) in `styles.csv`.

Rather than using the full dataset, a curated subset of approximately 6,000 images was selected across 10 categories, intentionally chosen to include semantically overlapping pairs such as casual shoes and sports shoes, and ethnic and non-ethnic wear. Confusable categories bring up more interesting failure cases during evaluation than a dataset containing perfectly categorizable data would.

- **Categories used:** Casual Shoes, Sports Shoes, Heels, Sandals, Kurtas (ethnic wear), Tshirts, Shirts, Tops, Handbags, Backpacks
- **Subset size:** approximately 6,000 images (capped per category to avoid category dominance)
- **Evaluation factor:** `articleType` labels from `styles.csv` are used to check whether search results match the query's expected category

## Architecture / Pipeline

**Offline:**
```
styles.csv + images  →  curation of subset  →  CLIP image encoder  →  image embeddings  →  FAISS index
```

**Online:**
```
text query  →  CLIP text encoder  →  text embedding  →  FAISS nearest neighbor search  →  ranked image results
```

Because CLIP's encoders are trained to place semantically similar concepts in the same vector space, a text embedding and an image embedding can be compared directly. The closer two vectors are (via cosine similarity), the more related their content. FAISS handles the actual nearest neighbor search.

The production app (`app.py`) uses CLIP ViT-B/32 for the live search experience. See the [Experiment section](#experiment-vit-b32-vs-vit-l14) for why this model was chosen over the larger ViT-L/14 variant.

## Why This Project

Retrieval and recommendation systems are core to search and discovery products like Copilot, Google Search, products at Meta (feed ranking), and Pinterest (visual search). This project explores that same underlying pattern, semantic retrieval via embeddings, applied to a concrete, visual domain.

This project works with pretrained transformer-based embeddings (CLIP), nearest-neighbor search (FAISS), and a real evaluation methodology (Precision@k). The ViT-B/32 vs. ViT-L/14 experiment specifically explores the tradeoff between model scale, retrieval accuracy, and query latency. It is a practical consideration in any production search system, where "bigger model" doesn't automatically mean "better results."

## Setup / How to Run

**1. Clone the repo and set up a virtual environment**
```bash
git clone https://github.com/tangopsd/muse-wardrobe-search-engine.git
cd muse-wardrobe-search-engine
python3 -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS/Linux
```

**2. Install dependencies**
```bash
pip install torch torchvision open_clip_torch faiss-cpu pandas numpy pillow streamlit
```

**3. Download the dataset**

Download [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) from Kaggle, and place it inside the project as:
```
data/images/
data/styles.csv
```

**4. Curate the dataset subset**
```bash
python curate_data.py
```

**5. Generate CLIP embeddings**
```bash
python generate_embeddings.py
```
This encodes the curated images with CLIP ViT-B/32 and saves the resulting vectors. Expect this step to take a while on CPU (no GPU required).

**6. Run the app**
```bash
streamlit run app.py
```
This opens the search interface in your browser at `http://localhost:8501`.

## Experiment: ViT-B/32 vs ViT-L/14

To understand how model size affects retrieval quality, the full pipeline was run twice: once with CLIP's smaller ViT-B/32 image encoder, and once with the larger ViT-L/14. Both were evaluated on the same 11 test queries, using Precision@5 and Precision@10 as the quality metric, and average query time as the speed metric.

<img src="assets/precision_comparison.png" width="500">

| Metric | ViT-B/32 | ViT-L/14 |
|---|---|---|
| Mean Precision@5 | 0.873 | 0.855 |
| Mean Precision@10 | 0.882 | 0.800 |
| Avg query time | 0.051s | 0.166s |

The larger model was consistently slower, roughly 3.2x the query latency of ViT-B/32, but it did not improve retrieval quality. On average, ViT-B/32 slightly outperformed ViT-L/14 on both Precision@5 and Precision@10.

<img src="assets/precision_best_worst.png" width="500">

Looking at individual queries tells a more nuanced story than the averages alone. Both models performed identically on the clearest queries (leather handbag, school backpack), where category boundaries are unambiguous. The gap shows up on harder, more visually confusable queries: "casual sneakers" improved slightly with ViT-L/14 at Precision@5, but dropped at Precision@10, while "summer sandals" (the weakest query for both models) failed completely under ViT-L/14, dropping from an already-poor 0.20 to 0.00.

<img src="assets/precision_all_queries.png" width="600">

**Takeaway:** on this dataset, a larger CLIP model did not translate to better retrieval quality, while costing over 3x the query latency. This suggests model size should be chosen based on dataset scale and complexity, not assumed to scale with quality, a genuinely useful engineering lesson for building retrieval systems in practice.

## Evaluation: Failure Cases

Beyond aggregate Precision@k scores, looking at actual search results reveals why certain queries underperform. Four representative queries were selected: two of the strongest results, and two of the weakest, shown here for both ViT-B/32 and ViT-L/14.

**Strong results**

*Leather handbag, ViT-B/32:*
<img src="assets/eval_leather_handbag_1.png" width="400">

<img src="assets/eval_leather_handbag_2.png" width="400">

*Leather handbag, ViT-L/14:*
<img src="assets/eval_leather_handbag_3.png" width="400">

<img src="assets/eval_leather_handbag_4.png" width="400">

"Leather handbag" (Precision@5 = 1.00 on both models) returns exclusively correct matches, unsurprising given handbags are visually and semantically distinct from every other category in the curated dataset.

*Ethnic kurta, ViT-B/32:*
<img src="assets/eval_ethnic_kurta_1.png" width="400">

<img src="assets/eval_ethnic_kurta_2.png" width="400">

*Ethnic kurta, ViT-L/14:*
<img src="assets/eval_ethnic_kurta_3.png" width="400">

<img src="assets/eval_ethnic_kurta_4.png" width="400">

"Ethnic kurta" (Precision@5 = 1.00 on ViT-B/32, 0.80 on ViT-L/14) performs well on both, correctly surfacing kurtas even without the word "kurta" appearing in most product names, evidence that the search is matching by visual/semantic concept rather than keyword.

**Weak results**

*Casual sneakers, ViT-B/32:*
<img src="assets/eval_casual_sneakers_1.png" width="400">

<img src="assets/eval_casual_sneakers_2.png" width="400">

*Casual sneakers, ViT-L/14:*
<img src="assets/eval_casual_sneakers_3.png" width="400">

<img src="assets/eval_casual_sneakers_4.png" width="400">

"Casual sneakers" (Precision@5 = 0.60 on ViT-B/32, 0.80 on ViT-L/14) shows visible confusion between Casual Shoes and Sports Shoes on both models. This is expected: both categories share a `subCategory` of "Shoes" and are visually similar (white sneaker silhouettes), which is exactly why this pair was deliberately included in the curated dataset as a confusable category test.

*Summer sandals, ViT-B/32:*
<img src="assets/eval_summer_sandals_1.png" width="400">

<img src="assets/eval_summer_sandals_2.png" width="400">

*Summer sandals, ViT-L/14:*
<img src="assets/eval_summer_sandals_3.png" width="400">

<img src="assets/eval_summer_sandals_4.png" width="400">

"Summer sandals" (Precision@5 = 0.20 on ViT-B/32, 0.00 on ViT-L/14) is the weakest query in the entire evaluation, and the only one where the larger model fails completely. This failure appears driven less by category overlap and more by the word "summer" itself, likely pulling in seasonally-associated but categorically unrelated items (e.g., other warm-weather apparel) rather than footwear specifically.

**Takeaway:** the dataset's deliberately curated confusable categories (Casual vs. Sports Shoes) surface exactly the kind of failure expected: near-misses within a visually similar category. The sandals failure is a different, arguably more interesting failure mode, showing that descriptive modifiers ("summer," "cozy," "oversized") can pull the search away from the intended object category entirely, a limitation worth flagging for anyone building a production system on top of this approach.

## Precision@k

Precision@k measures, out of the top k search results, how many are actually relevant. Since this dataset has no pre-existing relevance judgments for search queries, `articleType` labels from `styles.csv` were used as a proxy for ground truth: a result counts as relevant if its category matches what the query was expected to return (e.g., searching "sports shoes" and getting back a result labeled `articleType: Sports Shoes`).

Precision@5 and Precision@10 were computed for 11 test queries, each mapped to an expected category, and averaged across all queries.

<img src="assets/precision_all_queries.png" width="600">

| | ViT-B/32 | ViT-L/14 |
|---|---|---|
| Mean Precision@5 | 0.873 | 0.855 |
| Mean Precision@10 | 0.882 | 0.800 |

Full per-query results and model comparison are covered in the [Experiment section](#experiment-vit-b32-vs-vit-l14).
