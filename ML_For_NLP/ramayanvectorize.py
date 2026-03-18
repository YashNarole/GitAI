import re
import numpy as np
import os
from gensim.models import Word2Vec
try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("❌ Install: pip install pypdf")
    exit()

# ========================================
# CORRECTED PDF PATH (relative to script)
# ========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(script_dir, "Ramayana.of.Valmiki.by.Hari.Prasad.Shastri.pdf")

print("📖 Loading Ramayana...")
print(f"Script dir: {script_dir}")
print(f"PDF path: {pdf_path}")

# ========================================
# STEP 1: PDF LOADING ONLY
# ========================================
if not os.path.exists(pdf_path):
    print(f"❌ PDF NOT FOUND: {pdf_path}")
    print("💡 Put PDF in same folder as script")
    exit()

print("✅ PDF found, loading...")
try:
    reader = PdfReader(pdf_path)
    full_text = ""
    page_count = 0
    for page in reader.pages:
        text = page.extract_text()
        if text and len(text.strip()) > 20:
            full_text += text + " "
            page_count += 1
            if page_count > 200:  # Limit pages
                break
    print(f"✅ Loaded {page_count} pages, {len(full_text):,} chars")
except Exception as e:
    print(f"❌ PDF Error: {e}")
    print("💡 Check PDF is not corrupted/password protected")
    exit()

if len(full_text) < 1000:
    print("❌ PDF too empty - no text extracted")
    exit()

# ========================================
# STEP 2: PROCESS TEXT
# ========================================
print("🔧 Cleaning text...")
clean_text = re.sub(r'[^a-zA-Z\s]', ' ', full_text).lower()
sentences = re.split(r'[.!?]+', clean_text)
sentences = [s.strip().split() for s in sentences if len(s.strip()) > 10 and len(s.split()) > 3]

print(f"✅ {len(sentences)} sentences ready!")
if sentences:
    print("Sample:", ' '.join(sentences[0][:12]))
else:
    print("❌ NO SENTENCES CREATED!")
    exit()

# ========================================
# STEP 3: TRAIN WORD2VEC
# ========================================
print("\n🤖 Training Ramayana Word2Vec (2-5 mins)...")
model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=5,
    min_count=5,
    workers=4,
    epochs=20
)

##model.save("ramayana_word2vec.model")
##print("🎉 Model saved!")

# ========================================
# STEP 4: TEST THE MAGIC
# ========================================
print("\n🔥 TESTING RAMAYANA KNOWLEDGE:")
model = Word2Vec.load("ramayana_word2vec.model")

test_words = ['rama', 'lord', 'sita', 'ravana']
for word in test_words:
    if word in model.wv:
        print(f"\n📊 {word.upper()}'s closest neighbors:")
        for similar_word, score in model.wv.most_similar(word, topn=15):
            print(f"   {similar_word:<12} {score:.3f}")

print("\n📏 SIMILARITY SCORES:")
pairs = [('rama', 'lord'), ('rama', 'sita'), ('rama', 'killed'), ('rama', 'ayodhya'), ('rama', 'valmiki'), ('lanka', 'ayodhya')]
for w1, w2 in pairs:
    if w1 in model.wv and w2 in model.wv:
        sim = model.wv.similarity(w1, w2)
        print(f"  {w1:<6} ↔ {w2:<6} = {sim:.3f}")

print("\n🎊 SUCCESS! Word2Vec learned Ramayana relationships!")
print("💾 Model: 'ramayana_word2vec.model'")
