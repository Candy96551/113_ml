import os, shutil
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 類別與訓練資料（簡化）
docs = ["課表內容", "保險合約條款", "照片描述"]
labels = ["教育", "財務", "照片"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
clf = MultinomialNB().fit(X, labels)

def classify_text(text):
    vec = vectorizer.transform([text])
    return clf.predict(vec)[0]

def extract_text_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "".join([page.extract_text() or '' for page in pdf.pages])

def auto_sort(folder):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.endswith(".pdf"):
            text = extract_text_from_pdf(path)
            category = classify_text(text)
            dest = os.path.join("已分類", category)
            os.makedirs(dest, exist_ok=True)
            shutil.move(path, os.path.join(dest, file))
            print(f"{file} 分類為 {category}")

auto_sort("待分類")
