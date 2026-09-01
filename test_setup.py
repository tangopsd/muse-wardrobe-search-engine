import torch
import open_clip
import faiss
import pandas as pd
from PIL import Image

print("PyTorch:", torch.__version__)
print("FAISS OK")
print("open_clip models available:", open_clip.list_pretrained()[:3])