# ===============================
# 1. Import thư viện
# ===============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ===============================
# 2. Đọc dữ liệu
# ===============================
df = pd.read_excel("train_decision_tree_940cmt.xlsx")

# X: nội dung comment
X = df["comment_final"]

# y: nhãn sentiment
y = df["label"]


# ===============================
# 3. Chia dữ liệu 80% train - 20% test (Stratified)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", len(X_train))  # 752
print("Test size :", len(X_test))   # 188


# ===============================
# 4. Vector hóa văn bản (TF-IDF)
# ===============================
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)


# ===============================
# 5. Huấn luyện mô hình Decision Tree
# ===============================
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10
)

model.fit(X_train_tfidf, y_train)


# ===============================
# 6. Dự đoán & đánh giá mô hình
# ===============================
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy, 4))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ===============================
# 7. (Tuỳ chọn) Lưu kết quả dự đoán
# ===============================
result_df = pd.DataFrame({
    "comment": X_test,
    "sentiment_true": y_test,
    "sentiment_pred": y_pred
})

result_df.to_excel("sentiment_test_result.xlsx", index=False)
