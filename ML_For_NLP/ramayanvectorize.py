# PDF → BOW VECTORIZATION - END-TO-END (20 LINES)
import pandas as pd
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np  # ADD THIS LINE (line 1-2)


# 1. YOUR PDF PATH
pdf_path = r"C:\Users\PARAM 790\AI Playground\GitAI\ML_For_NLP\Ramayana.of.Valmiki.by.Hari.Prasad.Shastri-18-24.pdf"

# 2. EXTRACT TEXT FROM PDF
print("📄 Extracting text from Ramayana PDF...")
reader = PdfReader(pdf_path)
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + " "

print(f"✅ Extracted {len(full_text)} characters")

# 3. PREPROCESS
clean_text = re.sub('[^a-zA-Z]', ' ', full_text).lower()
corpus = [' '.join(clean_text.split())]  # Single document

# 4. VECTORIZE (BAG OF WORDS)
print("🎒 Creating Bag of Words...")
cv = CountVectorizer(max_features=2500, stop_words='english')
bow = cv.fit_transform(corpus).toarray()

# 5. RESULTS
print(f"✅ BOW Vector Shape: {bow.shape}")
print(f"✅ Vocabulary size: {len(cv.get_feature_names_out())}")
print("\nTop 10 words:")
print(pd.Series(bow[0]).sort_values(ascending=False).head(10))
print("\nFirst 20 words:", cv.get_feature_names_out()[:120])


# COMPLETE PDF VECTOR - FULL LIST (Run after previous code)

# Get the entire BOW vector as list
full_vector = bow[0].tolist()  # Convert numpy array → Python list

print("📋 COMPLETE PDF VECTOR (first 100 values):")
print(full_vector[:100])

print(f"\n📊 Full vector length: {len(full_vector)}")
print(f"Non-zero words (actual content): {(bow[0] > 0).sum()}")

# Show words with non-zero values (actual Ramayana content)
non_zero_indices = np.where(bow[0] > 0)[0]
print("\n🔤 Words found in Ramayana PDF:")
for idx in non_zero_indices[:20]:  # First 20 non-zero words
    word = cv.get_feature_names_out()[idx]
    count = int(bow[0][idx])
    print(f"  {word}: {count}")

# Save complete vector
import pickle
with open('ramayana_complete_vector.pkl', 'wb') as f:
    pickle.dump({
        'vector': full_vector,
        'vocabulary': cv.get_feature_names_out().tolist(),
        'non_zero_count': (bow[0] > 0).sum()
    }, f)

print("\n💾 Saved complete vector: ramayana_complete_vector.pkl")


