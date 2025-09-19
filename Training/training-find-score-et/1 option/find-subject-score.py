import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import re

from pathlib import Path

st.set_page_config(page_title="Dự đoán điểm môn", layout="wide")
st.title("📘 Dự đoán điểm số học phần ")


# Đường dẫn tới file mẫu có sẵn
template_path = Path("input-score.xlsx")

st.markdown("### 📥 Tải mẫu file nhập điểm có sẵn")
if template_path.exists():
    with open(template_path, "rb") as f:
        st.download_button(
            label="Tải xuống input-score.xlsx",
            data=f.read(),
            file_name="input-score.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.error(f"Không tìm thấy file mẫu tại {template_path}. Đặt `input-score.xlsx` vào thư mục chạy app.")


# -------------------------
# Helpers
# -------------------------
@st.cache_resource
def load_rf_model(target_name: str) -> object:
    # sanitize giống khi bạn lưu
    safe_name = re.sub(r'[\\/:\"*?<>| ]+', "_", target_name).lower()
    model_path = Path("models_streamlit") / f"rf_model_{safe_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model cho '{target_name}' tại {model_path}")
    return joblib.load(model_path)

LETTER_TO_GPA = {
    "A+": 4.0,
    "A": 4.0,
    "B+": 3.5,
    "B": 3.0,
    "C+": 2.5,
    "C": 2.0,
    "D+": 1.5,
    "D": 1.0,
}

def convert_letter_to_score(letter: str):
    if pd.isna(letter):
        return np.nan
    letter = letter.strip().upper()
    return LETTER_TO_GPA.get(letter, np.nan)

def build_feature_vector(df_input: pd.DataFrame, feature_order: list):
    """
    df_input: DataFrame with columns 'Môn học' and 'Điểm chữ' (or already numeric)
    feature_order: list of subject names in the same order the model expects
    """
    # map letter to numeric
    df_input["score"] = df_input["Điểm chữ"].apply(convert_letter_to_score)

    # collect in order; if missing subject, fill with nan
    features = []
    for subj in feature_order:
        # try exact match, fallback case-insensitive
        row = df_input[df_input["Môn học"].str.strip().str.lower() == subj.strip().lower()]
        if not row.empty:
            val = row.iloc[0]["score"]
        else:
            val = np.nan
        features.append(val)
    features = np.array(features).reshape(1, -1)
    return features

def infer_feature_order(example_model):
    """
    Heuristic: if the model has a .feature_names_in_ attribute (sklearn >=1.0), use it.
    Otherwise user must supply feature order manually.
    """
    if hasattr(example_model, "feature_names_in_"):
        return list(example_model.feature_names_in_)
    else:
        return None

# -------------------------
# UI: Upload + chọn mục tiêu
# -------------------------
st.sidebar.header("1. Tải file điểm lên")
uploaded = st.sidebar.file_uploader("Chọn file Excel đầu vào theo mẫu input-score.xlsx", type=["xlsx", "xls"])

st.sidebar.header("2. Chọn môn cần dự đoán")
target_subject = st.sidebar.selectbox("Môn học muốn dự đoán", [
    "Giải tích II", "Giải tích I", "Phương pháp tính", "Đại số", "Giải tích III",
    "Xác suất thống kê", "Vật lý đại cương II", "Vật lý đại cương I", "Tin học đại cương", "Vật lý điện tử",
    "Nhập môn kỹ thuật điện tử-viễn thông",
    "Thực tập cơ bản",
    "Technical Writing and Presentation",
    "Kỹ thuật lập trình C/C++",
    "Cấu kiện điện tử",
    "Lý thuyết mạch",
    "Tín hiệu và hệ thống",
    "Lý thuyết thông tin",
    "Cơ sở kỹ thuật đo lường",
    "Cấu trúc dữ liệu và giải thuật",
    "Trường điện từ",
    "Điện tử số",
    "Điện tử tương tự I",
    "Điện tử tương tự II",
    "Thông tin số",
    "Kỹ thuật phần mềm ứng dụng",
    "Anten và truyền sóng",
    "Đồ án thiết kế I",
    "Kỹ thuật vi xử lý",
    "Đồ án thiết kế II",
    "Xử lý tín hiệu số",
])

do_predict = st.sidebar.button("Dự đoán")

if uploaded is None:
    st.warning("Vui lòng tải lên file Excel chứa các môn và điểm chữ."); st.stop()

try:
    df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Không thể đọc file: {e}"); st.stop()

required_cols = ["Môn học", "Điểm chữ"]
if not all(col in df_raw.columns for col in required_cols):
    st.error(f"File đầu vào phải có ít nhất các cột: {required_cols}"); st.stop()

st.subheader("✅ Dữ liệu đã upload")
st.dataframe(df_raw[required_cols].head(50))

if not do_predict:
    st.info("Chọn môn và nhấn 'Chạy dự đoán' ở sidebar."); st.stop()

# -------------------------
# Load model
# -------------------------
try:
    model = load_rf_model(target_subject)
except FileNotFoundError as e:
    st.error(str(e)); st.stop()
except Exception as e:
    st.error(f"Lỗi khi load model: {e}"); st.stop()

# -------------------------
# Xây feature vector
# -------------------------
# Cố gắng đoán thứ tự feature từ model nếu có
feature_order = infer_feature_order(model)
if feature_order is None:
    st.warning(
        "Mô hình không có thông tin thứ tự biến (feature_names_in_); "
        "Bạn cần cung cấp thủ công danh sách các môn làm input đúng thứ tự huấn luyện."
    )
    st.info("Hiện tại dùng danh sách theo thứ tự trên file upload, sẽ điền theo thứ tự xuất hiện.")
    # fallback: lấy theo thứ tự xuất hiện trong file
    feature_order = df_raw["Môn học"].dropna().astype(str).tolist()

# Build vector
X = build_feature_vector(df_raw, feature_order)

# Kiểm tra missing
# if np.isnan(X).any():
#     st.warning("Có giá trị thiếu trong feature vector (một vài môn không có điểm chữ hợp lệ). Model có thể dự đoán kém hơn.")
    # # hiển thị môn thiếu
    # missing_idx = np.where(np.isnan(X.flatten()))[0]
    # missing_subjects = [feature_order[i] for i in missing_idx]
    # st.write("Thiếu điểm chữ cho các môn:", missing_subjects)

# -------------------------
# Dự đoán
# -------------------------
# try:
#     pred = model.predict(X)[0]
#     st.subheader(f"🎯 Kết quả dự đoán cho môn **{target_subject}**")
#     st.success(f"Điểm số dự đoán (ở dạng liên tục): {pred:.3f}")
# except Exception as e:
#     st.error(f"Không thể dự đoán: {e}")
#     st.stop()

# # Nếu bạn muốn chuyển về điểm chữ có thể quy ngược:
# def gpa_to_letter(gpa_value):
#     # đơn giản mapping ngược gần đúng
#     if gpa_value >= 3.75:
#         return "A / A+"
#     if gpa_value >= 3.25:
#         return "B+"
#     if gpa_value >= 2.75:
#         return "B"
#     if gpa_value >= 2.25:
#         return "C+"
#     if gpa_value >= 1.75:
#         return "C"
#     if gpa_value >= 1.25:
#         return "D+"
#     return "D or below"

# st.info(f"Quy ước ra điểm chữ: {gpa_to_letter(pred)}")

# # -------------------------
# # Tùy chọn: hiển thị feature đầu vào
# # -------------------------
# with st.expander("🔍 Xem vector đầu vào (features) gửi cho mô hình"):
#     df_features = pd.DataFrame([X.flatten()], columns=feature_order)
#     st.dataframe(df_features.T.rename(columns={0: "Giá trị"}))


num_valid = np.count_nonzero(~np.isnan(X.flatten()))
if num_valid < 5:
    st.error(f"Không đủ dữ liệu để dự đoán: chỉ có {num_valid} môn hợp lệ, cần ít nhất 5 môn."); st.stop()


try:
    pred = model.predict(X)[0]
    st.subheader(f"🎯 Kết quả dự đoán cho môn **{target_subject}**")
    # st.success(f"Điểm số (liên tục) dự đoán: {pred:.3f}")
except Exception as e:
    st.error(f"Không thể dự đoán: {e}")
    st.stop()

# -------------------------
# Quy đổi sang điểm chữ rồi về điểm số chuẩn
# -------------------------
def numeric_to_letter(score: float) -> str:
    if score >= 3.75:
        return "A / A+"
    if score >= 3.25:
        return "B+"
    if score >= 2.75:
        return "B"
    if score >= 2.25:
        return "C+"
    if score >= 1.75:
        return "C"
    if score >= 1.25:
        return "D+"
    return "D"

LETTER_TO_NUMERIC = {
    "A+": 4.0,
    "A": 4.0,
    "B+": 3.5,
    "B": 3.0,
    "C+": 2.5,
    "C": 2.0,
    "D+": 1.5,
    "D": 1.0,
}

# Lấy điểm chữ gần đúng
letter = numeric_to_letter(pred)
# Nếu trả về "A / A+" thì giữ thành A (vì cả hai quy về 4.0)
base_letter = "A" if "A" in letter else letter  # xử lý "A / A+"
converted_numeric = LETTER_TO_NUMERIC.get(base_letter, np.nan)

# st.subheader("🔁 Quy đổi chuẩn")
st.success(f"- **Điểm chữ dự đoán:** {letter}")
st.success(f"- **Điểm số chuẩn tương ứng:** {converted_numeric:.1f}")