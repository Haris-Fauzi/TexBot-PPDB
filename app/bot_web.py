from flask import Flask, request, render_template, session, redirect, url_for
from flask_session import Session
import os

import google.generativeai as genai
from joblib import Memory
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from chromadb import PersistentClient

# --- Flask setup ---
app = Flask(__name__)
app.secret_key = "rahasia123"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# --- Gemini setup ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyBRAQhHXrwkh9C9rXEHQ7vPoOPbIiTyhc0"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-2.5-flash')

# --- Embeddings + Chroma setup ---
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chroma_client = PersistentClient(path="D:/Data/Pelatihan Ai-DS/MagangAI/TexBot/chroma_db")
collection = chroma_client.get_collection(name="ppdb_faq")

THRESHOLD = 0.6

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        user_query = request.form["question"]

        # --- Cari di Chroma ---
        query_embedding = embedder.encode([user_query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=3)

        if results["documents"] and results["documents"][0]:
            doc_embeddings = embedder.encode(results["documents"][0])
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

            best_idx = int(np.argmax(similarities))
            best_score = similarities[best_idx]
            best_doc = results["documents"][0][best_idx]
            best_meta = results["metadatas"][0][best_idx]

            if best_score >= THRESHOLD:
                context_faq = f"""
                Pertanyaan FAQ: {best_doc}
                Jawaban FAQ: {best_meta.get('answer', '')}
                """

                prompt = f"""
                Anda adalah asisten PPDB SMK Texmaco Semarang.
                Gunakan informasi FAQ berikut untuk menjawab pertanyaan.

                ### FAQ:
                {context_faq}

                ### Pertanyaan pengguna:
                {user_query}

                Jawaban:
                """
                response = model.generate_content(prompt)
                answer = response.text.strip()
            else:
                answer = f"❌ Maaf, tidak ada jawaban relevan. (skor={round(float(best_score),4)})"
        else:
            answer = "❌ Maaf, tidak ada data FAQ ditemukan."

        # Simpan ke session
        session["chat_history"].append({"user": user_query, "bot": answer})
        session.modified = True

        # ⬅️ setelah POST, redirect ke GET biar aman saat refresh
        return redirect(url_for("index"))

    # GET request
    return render_template("index.html", chat_history=session["chat_history"])


@app.route("/reset", methods=["POST"])
def reset():
    session["chat_history"] = []
    session.modified = True
    # kalau pakai memory LangChain, sekalian clear juga:
    try:
        Memory.chat_memory.clear()
    except:
        pass
    # PENTING: redirect ke "/" supaya URL kembali ke halaman utama
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
