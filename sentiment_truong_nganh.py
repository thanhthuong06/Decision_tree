import pandas as pd
import os

# ==============================================================================
# PHẦN 1: ĐỌC FILE DỮ LIỆU
# ==============================================================================
input_file = "sentiment_pred.xlsx"
output_file = "Ket_qua_phan_tich_Chu_de_Truong_Nganh_pred.xlsx"

current_folder = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_folder, input_file)
output_path = os.path.join(current_folder, output_file)

try:
    print(f">>> Đang đọc file: {input_file}")
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Không tìm thấy file '{input_file}'")
    exit()

# ==============================================================================
# PHẦN 2: ĐỊNH NGHĨA CHỦ ĐỀ
# ==============================================================================
topics = {
    "Học phí & Chi phí": [
        "học phí", "tiền", "đắt", "rẻ", "chi phí", "đóng", "nộp", "hỗ trợ", "giảm", "lệ phí"
    ],
    "Bằng cấp & Giá trị": [
        "bằng", "chính quy", "xin việc", "giá trị", "tốt nghiệp", "liên thông", "công nhận"
    ],
    "Chương trình & Chất lượng": [
        "giáo trình", "tài liệu", "kiến thức", "dạy", "thầy", "cô", "giảng viên",
        "chất lượng", "khó", "dễ", "hay", "chán", "đuối"
    ],
    "Thời gian & Hình thức thi": [
        "thời gian", "lịch", "tối", "thứ 7", "chủ nhật", "online", "thi",
        "kiểm tra", "zoom", "trực tuyến"
    ],
    "Tư vấn & Thủ tục": [
        "tuyển sinh", "đăng ký", "hồ sơ", "xét tuyển", "tư vấn", "liên hệ", "inbox"
    ]
}

def detect_topic(text):
    text = str(text).lower()
    detected = []

    for topic, keywords in topics.items():
        for kw in keywords:
            if kw in text:
                detected.append(topic)
                break

    return ", ".join(detected) if detected else "Khác"

# ==============================================================================
# PHẦN 2.1: ĐỊNH NGHĨA TỪ KHÓA TÊN TRƯỜNG & NGÀNH
# ==============================================================================

schools = {
    "ĐH Kinh tế Quốc dân": ["kinh tế quốc dân", "neu"],
    "ĐH Thương mại": ["đại học thương mại", "tmu"],
    "ĐH Kinh tế TP.HCM": ["đại học kinh tế tphcm", "ueh"],
    "ĐH Mở Hà Nội": ["đại học mở hà nội", "hou"],
    "ĐH Mở TP.HCM": ["đại học mở tphcm", "ou"],
    "ĐH Hà Nội": ["đại học hà nội", "hanu"],
    "ĐH Kinh tế – Kỹ thuật Công nghiệp": ["kinh tế kỹ thuật công nghiệp", "uneti"],
    "Học viện Tài chính": ["học viện tài chính", "aof"],
    "Học viện Ngân hàng": ["học viện ngân hàng", "bav"],
    "Học viện CN Bưu chính Viễn thông": ["bưu chính viễn thông", "ptit"],
    "ĐH Ngân hàng TP.HCM": ["đại học ngân hàng", "hub"],
    "ĐH Hoa Sen": ["hoa sen", "hsu"],
    "ĐH Đại Nam": ["đại học đại nam", "dnu"],
    "ĐH Thái Nguyên": ["đại học thái nguyên", "thái nguyên", "tnu"]
}

majors = {
    "Công nghệ thông tin": ["cntt", "công nghệ thông tin", "it"],
    "Quản trị kinh doanh": ["qtkd", "quản trị kinh doanh"],
    "Marketing": ["marketing", "maketing", "mkt"],
    "Kế toán": ["kế toán", "ke toan"],
    "Kiểm toán": ["kiểm toán"],
    "Tài chính – Ngân hàng": ["tài chính ngân hàng", "tcnh"],
    "Ngôn ngữ Anh": ["ngôn ngữ anh", "anh văn", "nna"],
    "Thương mại điện tử": ["thương mại điện tử", "tmdt"],
    "Luật": ["luật"],
    "Luật kinh tế": ["luật kinh tế"],
    "Logistics": ["logistics"],
    "Quản trị nhân lực": ["quản trị nhân lực", "qtnl"],
    "Hệ thống thông tin quản lý": ["hệ thống thông tin", "httt", "mis"],
    "Đào tạo từ xa": ["đào tạo từ xa", "đttx"]
}

# ==============================================================================
# PHẦN 3: NHẬN DIỆN TÊN TRƯỜNG
# ==============================================================================

def extract_school(text):
    text = str(text).lower()
    found = []

    for school, keywords in schools.items():
        for kw in keywords:
            if kw in text:
                found.append(school)
                break

    return ", ".join(found) if found else "Không đề cập"

# ==============================================================================
# PHẦN 4: NHẬN DIỆN TÊN NGÀNH
# ==============================================================================

def extract_major(text):
    text = str(text).lower()
    found = []

    for major, keywords in majors.items():
        for kw in keywords:
            if kw in text:
                found.append(major)
                break

    return ", ".join(found) if found else "Không đề cập"

# ==============================================================================
# PHẦN 5: TIỀN XỬ LÝ & GẮN NHÃN
# ==============================================================================

print(">>> Đang xử lý dữ liệu...")

if 'comment_final' not in df.columns:
    df['comment_final'] = df['Binh_luan_goc'].astype(str).str.lower()

df['Chu_de'] = df['comment_final'].apply(detect_topic)
df['Ten_truong'] = df['comment_final'].apply(extract_school)
df['Ten_nganh'] = df['comment_final'].apply(extract_major)

# ==============================================================================
# PHẦN 6: LƯU FILE
# ==============================================================================
df.to_excel(output_path, index=False)
print(f"Đã lưu kết quả: {output_file}")

# ==============================================================================
# PHẦN 7: BÁO CÁO NHANH
# ==============================================================================
print("\n" + "="*50)
print("THỐNG KÊ CHỦ ĐỀ")
print("="*50)
print(df['Chu_de'].str.split(', ').explode().value_counts())

print("\n" + "="*50)
print("TOP TRƯỜNG ĐƯỢC NHẮC NHIỀU")
print("="*50)
print(df['Ten_truong'].value_counts().head(10))

print("\n" + "="*50)
print("TOP NGÀNH ĐƯỢC NHẮC NHIỀU")
print("="*50)
print(df['Ten_nganh'].value_counts().head(10))

if 'Nhan_chu' in df.columns:
    print("\n" + "="*50)
    print("CẢM XÚC THEO CHỦ ĐỀ")
    print("="*50)
    df_ex = df.assign(Chu_de_don=df['Chu_de'].str.split(', ')).explode('Chu_de_don')
    print(pd.crosstab(df_ex['Chu_de_don'], df_ex['Nhan_chu']))

print("\nHOÀN THÀNH!")
