import os
import logging
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from chromadb import PersistentClient  # Mengganti Client dengan PersistentClient
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

# --- Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- Setup Gemini ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyBRAQhHXrwkh9C9rXEHQ7vPoOPbIiTyhc0"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- Setup Embedding + Chroma ---
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # dim=384

# Perbaikan: Gunakan PersistentClient dengan path yang benar
chroma_client = PersistentClient(path="D:/Data/Pelatihan Ai-DS/MagangAI/TexBot/chroma_db")

# Pastikan collection ada, jika tidak buat yang baru
try:
    collection = chroma_client.get_collection(name="ppdb_faq")
    print("Collection ppdb_faq ditemukan!")
except:
    # Jika collection tidak ditemukan, buat yang baru
    print("Collection ppdb_faq tidak ditemukan, membuat yang baru...")
    collection = chroma_client.create_collection(name="ppdb_faq")

THRESHOLD = 0.6

# --- Telegram Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Halo! Saya adalah bot PPDB SMK Texmaco Semarang.\n"
        "Kirim pertanyaan Anda, dan saya akan mencoba menjawab berdasarkan FAQ."
    )

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    try:
        # Encode pertanyaan user
        query_embedding = embedder.encode([user_query]).tolist()  # Konversi ke list untuk ChromaDB
        
        # Cari 3 dokumen terdekat
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )
        
        # Periksa apakah ada hasil yang ditemukan
        if not results["documents"] or not results["documents"][0]:
            await update.message.reply_text("❌ Maaf, tidak ada data FAQ yang ditemukan.")
            return
            
        # Encode ulang dokumen hasil query untuk cek similarity
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

            formatted_prompt = f"""
            Anda adalah asisten PPDB SMK Texmaco Semarang.
            Gunakan informasi FAQ yang disediakan untuk menjawab pertanyaan dengan jelas.
            Jika informasi tidak relevan, katakan "Maaf, saya tidak memiliki informasi yang relevan."

            ### FAQ Terkait:
            {context_faq}

            ### Pertanyaan pengguna:
            {user_query}

            Jawaban:
            """
            response = model.generate_content(formatted_prompt)
            final_answer = response.text.strip()

            reply_text = f"{final_answer}\n\n(Similarity: {round(float(best_score), 4)})"
        else:
            reply_text = f"❌ Maaf, saya belum punya jawaban yang relevan.\nSkor tertinggi hanya: {round(float(best_score),4)}"

        await update.message.reply_text(reply_text)

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)  # tampilkan stacktrace
        await update.message.reply_text(f"❌ Maaf, error: {e}")


# --- Main Bot ---
if __name__ == "__main__":
    TELEGRAM_TOKEN = "8447704401:AAF38uYTnWXwdYv3IXKX8tiw2JRwKVpSon0"

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    print("Bot Telegram siap dijalankan...")
    app.run_polling()