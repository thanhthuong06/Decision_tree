import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import joblib

# ===============================
# Đọc dữ liệu đã gán nhãn (940 cmt)
# ===============================
train_df = pd.read_excel("train_decision_tree_940cmt.xlsx")

X = train_df["comment_final"]
y = train_df["label"]

# Vector hóa
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1,2)
)

X_tfidf = tfidf.fit_transform(X)

# Train model
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,
    class_weight="balanced"  # QUAN TRỌNG
)

model.fit(X_tfidf, y)

# Lưu model & vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Đã train và lưu mô hình")
