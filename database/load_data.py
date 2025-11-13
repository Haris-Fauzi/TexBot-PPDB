#Buat Database Vector menggunakan Chroma
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# --- 1. Load CSV ---
df = pd.read_csv("../dataset/enhanced_ppdb_faq_dataset.csv")

print("Jumlah data:", len(df))

# --- 2. Inisialisasi Chroma + Embedding ---
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # dim=384
chroma_client = chromadb.PersistentClient(path="../chroma_db")


# Buat collection baru (hapus dulu kalau sudah ada)
try:
    chroma_client.delete_collection("ppdb_faq")
except:
    pass

collection = chroma_client.create_collection("ppdb_faq")

# --- 3. Insert data ke Chroma ---
for idx, row in df.iterrows():
    q = row["question"]
    a = row["answer"]

    emb = embedder.encode(q).tolist()
    collection.add(
        ids=[f"faq-{idx}"],
        embeddings=[emb],
        documents=[q],
        metadatas=[{"answer": a}]
    )

print("✅ Dataset berhasil dimasukkan ke Chroma!")
