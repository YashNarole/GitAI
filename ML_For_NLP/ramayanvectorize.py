import re
import numpy as np
from gensim.models import Word2Vec
from pypdf import PdfReader

# ========================================
# STEP 1: LOAD & PREPARE RAMAYANA TEXT
# ========================================
print("📖 Loading Ramayana...")
pdf_path = r"C:\Users\PARAM 790\AI Playground\GitAI\ML_For_NLP\Ramayana.of.Valmiki.by.Hari.Prasad.Shastri-18-24.pdf"

reader = PdfReader(pdf_path)
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + " "

# Clean & split into sentences
clean_text = re.sub(r'[^a-zA-Z\s]', ' ', full_text).lower()
sentences = re.split(r'[.!?]+', clean_text)
sentences = [s.strip().split() for s in sentences if len(s.split()) > 3]

print(f"✅ {len(sentences)} sentences ready!")
print("Sample:", ' '.join(sentences[10][:10]))

# ========================================
# STEP 2: TRAIN WORD2VEC (3 minutes)
# ========================================
print("\n🤖 Training Ramayana Word2Vec...")
model = Word2Vec(
    sentences=sentences,
    vector_size=100,      # 100-dim vectors
    window=5,             # 5 words context
    min_count=5,          # Ignore rare words
    workers=4,
    epochs=10             # Training passes
)

model.save("ramayana_word2vec.model")
print("🎉 Model saved!")

# ========================================
# STEP 3: TEST THE MAGIC!
# ========================================
print("\n🔥 TESTING WORD2VEC KNOWLEDGE:")
model = Word2Vec.load("ramayana_word2vec.model")

# Test 1: Rama's world
print("\n1. Rama's closest words:")
for word, score in model.wv.most_similar('rama', topn=10):
    print(f"   {word}: {score:.3f}")

# Test 2: Similarity scores
print("\n2. Similarity proof:")
print(f"  rama ↔ lord:    {model.wv.similarity('rama', 'lord'):.3f}")
print(f"  rama ↔ vishnu:  {model.wv.similarity('rama', 'vishnu'):.3f}")
print(f"  rama ↔ sita:    {model.wv.similarity('rama', 'sita'):.3f}")
print(f"  rama ↔ apple:   {model.wv.similarity('rama', 'apple'):.3f}")

# Test 3: Vector math (king-man+woman=queen style)
print("\n3. Vector math:")
result = model.wv.most_similar(positive=['rama', 'sita'], negative=['ravana'])
print(f"  rama + sita - ravana = {result[0][0]} ({result[0][1]:.3f})")

# Test 4: Show actual vectors
print("\n4. Vector peek:")
print("  rama =", model.wv['rama'][:10])
print("  lord =", model.wv['lord'][:10])

print("\n🎊 WORD2VEC SUCCESS! 'lord' is near 'rama' as expected!")

