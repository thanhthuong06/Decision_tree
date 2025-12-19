import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================================================================
# PHẦN 1: CẤU HÌNH & ĐỌC FILE
# ==============================================================================
# Tên file Excel gốc của bạn
file_input_name = "_[NCKH] Data final.xlsx"
# Tên file kết quả sẽ lưu ra
file_output_name = "Ket_qua_phan_tich.xlsx"

# Lấy đường dẫn thư mục hiện tại
current_folder = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(current_folder, file_input_name)
output_path = os.path.join(current_folder, file_output_name)

print(f">>> Đang đọc dữ liệu từ: {file_input_name}")

texts_fb = []
texts_tiktok = []

try:
    if not os.path.exists(excel_path):
        print(f"LỖI: Không tìm thấy file '{file_input_name}'. Hãy để file Excel cùng thư mục với code này.")
        exit()

    # Đọc FB
    df_fb = pd.read_excel(excel_path, sheet_name='FB + threads', engine='openpyxl')
    if 'comment_final' in df_fb.columns:
        texts_fb = df_fb['comment_final'].dropna().astype(str).tolist()
    
    # Đọc Tiktok
    df_tiktok = pd.read_excel(excel_path, sheet_name='Tiktok', header=None, engine='openpyxl')
    if df_tiktok.shape[1] > 1:
        texts_tiktok = df_tiktok.iloc[:, 1].dropna().astype(str).tolist()

except Exception as e:
    print(f"Lỗi đọc file: {e}")
    exit()

# Gộp dữ liệu
all_texts = texts_fb + texts_tiktok
df = pd.DataFrame({'Binh_luan_goc': all_texts})
print(f"Tổng số dòng dữ liệu: {len(df)}")

# ==============================================================================
# PHẦN 2: XỬ LÝ & GÁN NHÃN
# ==============================================================================
print("- Đang xử lý và gán nhãn...")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Clean_text'] = df['Binh_luan_goc'].apply(clean_text)

# Từ khóa
pos_keywords = ['tốt', 'ok', 'hay', 'thích', 'tuyệt', 'ổn', 'giỏi', 'ngon', 'giá trị', 'hợp lý', 'uy tín', 'xịn']
neg_keywords = ['kém', 'tệ', 'chán', 'đuối', 'khó', 'đắt', 'lừa', 'không tốt', 'mệt', 'phí', 'thất vọng']

def get_sentiment_auto(text):
    count_pos = sum([1 for word in pos_keywords if word in text])
    count_neg = sum([1 for word in neg_keywords if word in text])
    if count_pos > count_neg: return 1
    elif count_neg > count_pos: return 0
    else: return -1

df['Nhan_so'] = df['Clean_text'].apply(get_sentiment_auto)

# Tạo cột nhãn chữ cho dễ đọc trong báo cáo
def label_to_text(val):
    if val == 1: return "Tích cực"
    elif val == 0: return "Tiêu cực"
    else: return "Chưa xác định"

df['Nhan_chu'] = df['Nhan_so'].apply(label_to_text)

# Lọc dữ liệu có nhãn để train
df_model = df[df['Nhan_so'] != -1].copy()
print(f"Số lượng mẫu dùng để Train: {len(df_model)}")

# ==============================================================================
# PHẦN 3: HUẤN LUYỆN & ĐÁNH GIÁ (DECISION TREE)
# ==============================================================================
print("- Đang chạy mô hình Decision Tree...")
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
X = vectorizer.fit_transform(df_model['Clean_text'])
y = df_model['Nhan_so']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='gini', max_depth=30, random_state=42)
clf.fit(X_train, y_train)

# Đánh giá
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*30)
print(f"KẾT QUẢ ĐỘ CHÍNH XÁC: {acc:.2%}")
print("="*30)
print(classification_report(y_test, y_pred, target_names=['Tiêu cực', 'Tích cực']))

# ==============================================================================
# PHẦN 4: LƯU FILE KẾT QUẢ
# ==============================================================================
print(f"- Đang lưu file kết quả ra: {file_output_name}...")

try:
    # Xuất ra Excel
    # Chỉ lấy các cột cần thiết cho báo cáo
    df_export = df[['Binh_luan_goc', 'Nhan_chu', 'Nhan_so']]
    df_export.to_excel(output_path, index=False)
    
    print("\n" + "="*40)
    print("ĐÃ XONG! FILE ĐƯỢC LƯU TẠI:")
    print(output_path)
    print("="*40)
except Exception as e:
    print(f"Lỗi khi lưu file: {e}")
    print("Hãy tắt file Excel nếu bạn đang mở nó nhé.")