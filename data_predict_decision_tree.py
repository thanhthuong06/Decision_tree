import pandas as pd
import joblib

# ===============================
# Load model đã train
# ===============================
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ===============================
# Đọc dữ liệu CHƯA NHÃN (5700 cmt)
# ===============================
unlabel_df = pd.read_excel("Data_predict.xlsx")

X_new = unlabel_df["comment_final"]

# Vector hóa
X_new_tfidf = tfidf.transform(X_new)

# Dự đoán sentiment
unlabel_df["sentiment_pred"] = model.predict(X_new_tfidf)

# Lưu kết quả
unlabel_df.to_excel("sentiment_pred.xlsx", index=False)

print("Đã gán nhãn tự động cho 5700 comment")
