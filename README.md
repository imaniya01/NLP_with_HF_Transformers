# Natural Language Processing with Hugging Face Transformers 

<p>
  Repository ini berisi latihan mandiri mengenai <b>Natural Language Processing (NLP)</b> menggunakan Hugging Face Transformers.
</p>

---

### NLP Exercises

Di bawah ini adalah dokumentasi latihan yang telah saya kerjakan, mencakup kode implementasi, hasil output, dan analisis singkatnya.

#### 1. Exercise 1 - Sentiment Analysis
<p>
  Menggunakan model yang dilatih khusus pada tweet untuk menganalisis sentimen kalimat yang mengandung hashtag.
</p>

```python
# --- CODE ---
# Model: cardiffnlp/twitter-roberta-base-sentiment
specific_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
tweet = "I finally finished my project! It was hard work but the result is amazing. #proud"
results = specific_model(tweet)
print(results)

# --- RESULT ---
# [{'label': 'LABEL_2', 'score': 0.9850379824638367}]
# Note: LABEL_2 corresponds to Positive.

# --- ANALYSIS ---
# Model berhasil mengidentifikasi kalimat sebagai positif dengan skor keyakinan yang sangat tinggi (sekitar 98.5%). 
# Meskipun kalimat mengandung frasa "hard work" (yang bisa bermakna negatif), model dengan tepat menafsirkan 
# konteks keseluruhan dan hashtag "#proud" sebagai sentimen positif.

#### 2. Exercise 2 - Topic Classification

<p>Menggunakan <i>Zero-Shot Classification</i> untuk mengkategorikan kalimat tentang peraturan pajak ke dalam topik tertentu tanpa pelatihan sebelumnya. </p>

```python
# --- CODE ---
# Model: facebook/bart-large-mnli
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentence = "The government announced new tax regulations for digital businesses starting next year."
labels = ["politics", "economy", "sports", "health"]
result = classifier(sentence, candidate_labels=labels)
print(result)

# --- RESULT ---
# {
#   'sequence': 'The government announced new tax regulations for digital businesses starting next year.', 
#   'labels': ['economy', 'politics', 'health', 'sports'], 
#   'scores': [0.68, 0.20, 0.08, 0.04]
# }

# --- ANALYSIS ---
# Classifier dengan tepat mengidentifikasi "economy" sebagai topik yang paling relevan dengan skor tertinggi (~0.68), 
# diikuti oleh politics. Ini menunjukkan kemampuan model untuk memahami konteks kata kunci seperti "tax" dan "business" 
# dan memetakannya ke label kandidat yang diberikan.

#### 3. Exercise 3 - Text Generation Model

<p> Menggunakan model GPT-2 untuk melengkapi kalimat mengenai masa depan kecerdasan buatan (AI). </p>

```python
# --- CODE ---
# Model: gpt2
generator = pipeline('text-generation', model='gpt2')
start_sentence = "The future of Artificial Intelligence is"
generated_text = generator(start_sentence, max_length=50, num_return_sequences=3)

# --- RESULT (Variations) ---
# 1. "...to take the computer-based AI and combine it with the data collected from humans..."
# 2. "The future of Artificial Intelligence is certainly bright."
# 3. "...will become the most popular topic of discussion..."

# --- ANALYSIS ---
# Model GPT-2 menghasilkan kelanjutan kalimat yang koheren dan benar secara tata bahasa. 
# Model memberikan berbagai perspektif, mulai dari implementasi teknis (menggabungkan data) 
# hingga pernyataan optimis umum ("certainly bright").

#### 4. Exercise 4 - Name Entity Recognition (NER

<p> Mengekstrak entitas (Orang, Organisasi, Lokasi) dari kalimat biografi kustom. </p>

```python
# --- CODE ---
# Model: dbmdz/bert-large-cased-finetuned-conll03-english
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("My name is Imaniya, I am a student at UNESA, majoring in Informatics Engineering.")

# --- RESULT ---
# Imaniya: Identified as PER (Person)
# UNESA: Identified as ORG (Organization)

# --- ANALYSIS ---
# Model secara akurat mengekstrak "Imaniya" sebagai nama orang dan "UNESA" sebagai organisasi. 
# Hal ini menunjukkan kemampuan model untuk membedakan kata benda khusus (proper nouns) 
# dan mengklasifikasikannya dengan benar dalam konteks kalimat.

#### 5. Exercise 5 - Question Answering

<p> Mengekstrak jawaban langsung dari konteks teks mengenai Candi Borobudur. </p>

```python
# --- CODE ---
# Model: distilbert-base-cased-distilled-squad
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "Where is Borobudur located?"
context = """Borobudur is a 9th-century Mahayana Buddhist temple in Magelang Regency,
not far from the town of Muntilan, in Central Java, Indonesia.
It is the world's largest Buddhist temple"""
qa_model(question=question, context=context)

# --- RESULT ---
# {'score': ..., 'start': ..., 'end': ..., 'answer': 'Magelang Regency'}

# --- ANALYSIS ---
# Model dengan tepat menunjukkan lokasi spesifik "Magelang Regency" dari teks yang disediakan. 
# Model mengabaikan detail geografis tambahan untuk memberikan jawaban yang paling langsung 
# terhadap pertanyaan "Where".

#### 6. Exercise 6 - Text Summarization

<p> Meringkas paragraf teknis mengenai bahasa pemrograman Python. </p>

```python
# --- CODE ---
# Model: sshleifer/distilbart-cnn-12-6
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
long_text = """Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms..."""
summary = summarizer(long_text, max_length=60, min_length=30, do_sample=False)
print(summary[0]['summary_text'])

# --- RESULT ---
# "Python is a high-level, general-purpose programming language"

# --- ANALYSIS ---
# Summarizer berhasil meringkas paragraf menjadi definisi yang paling esensial. 
# Model menangkap materi inti secara ringkas, menghapus penjelasan mendetail untuk memenuhi batasan panjang teks.

#### 7. Exercise 7 - Translation

<p> Menerjemahkan kalimat bahasa Inggris ke bahasa Jerman menggunakan model T5. </p>

```python
# --- CODE ---
# Model: t5-small (translation_en_to_de)
translator = pipeline("translation_en_to_de", model="t5-small")
english_text = "I am learning how to use Hugging Face Transformers today."
german_translation = translator(english_text)
print(f"Jerman: {german_translation[0]['translation_text']}")

# --- RESULT ---
# Jerman: Ich lerne heute, wie man Hugging Face Transformers benutzt.

# --- ANALYSIS ---
# Model T5 berhasil menangani tugas penerjemahan dari Inggris ke Jerman, mengonversi struktur kalimat 
# ke bahasa target sambil tetap mempertahankan makna asli dan konteks teknisnya.

### Tools Used

<p>
  <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" alt="Python">
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab">
</p>
