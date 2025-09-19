import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, re
from pathlib import Path

# =========================
# UI / Layout (giữ nguyên)
# =========================
st.set_page_config(page_title="Dự đoán điểm môn", layout="wide")
st.title("📘 Dự đoán điểm số học phần ")

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

# =========================
# Artifacts (đường dẫn NHỚ như đã in)
# =========================
MODELS_DIR = Path("models_streamlit_xgb")            # chứa rf_model_{safe_name}.joblib
INDEX_CSV  = Path("index.csv")                       # sổ tra cứu target,K -> model_path (bạn lưu ngoài folder)
SCALER_P   = Path("2/scaler.joblib")                 # scaler chuẩn hoá
SUBJECTS_P = Path("3/subjects.json")                 # danh sách/trật tự môn
MF_PATH    = Path("models_streamlit_mf/find-subject-score.joblib")  # MF artifacts
GGM_PATH   = Path("models_streamlit_ggm/ggm.joblib")               # GGM artifacts (đã train và lưu tại đây)

# =========================
# Helpers (hybrid engine)
# =========================
def safe_name(text: str) -> str:
    return re.sub(r'[\\/:\"*?<>| ]+', "_", str(text)).strip("_").lower()

@st.cache_resource
def load_subjects_means_stds():
    subjects = json.loads(SUBJECTS_P.read_text(encoding="utf-8"))
    scaler   = joblib.load(SCALER_P)
    means = pd.Series(scaler["means"])
    stds  = pd.Series(scaler["stds"]).replace(0, 1.0)
    return subjects, means, stds

@st.cache_resource
def load_xgb_index():
    if INDEX_CSV.exists():
        return pd.read_csv(INDEX_CSV)
    return pd.DataFrame(columns=["target","K","model_path"])

@st.cache_resource
def load_mf():
    if MF_PATH.exists():
        return joblib.load(MF_PATH)
    return None

@st.cache_resource
def load_ggm():
    if GGM_PATH.exists():
        return joblib.load(GGM_PATH)
    return None

LETTER_TO_GPA = {
    "A+": 4.0, "A": 4.0,
    "B+": 3.5, "B": 3.0,
    "C+": 2.5, "C": 2.0,
    "D+": 1.5, "D": 1.0,
}
def convert_letter_to_score(letter: str):
    if pd.isna(letter): return np.nan
    return LETTER_TO_GPA.get(str(letter).strip().upper(), np.nan)

def numeric_to_letter(score: float) -> str:
    if score >= 3.75: return "A / A+"
    if score >= 3.25: return "B+"
    if score >= 2.75: return "B"
    if score >= 2.25: return "C+"
    if score >= 1.75: return "C"
    if score >= 1.25: return "D+"
    return "D"

def standardize_user_row(user_numeric: dict, subjects, means, stds):
    vals = []
    for s in subjects:
        v = user_numeric.get(s, np.nan)
        if pd.isna(v): vals.append(np.nan)
        else:          vals.append((float(v) - means[s]) / stds[s])
    return np.array(vals, dtype=float)

def build_masked_features(std_row: np.ndarray, kept_subjects: list, subjects):
    col_index = {s:i for i,s in enumerate(subjects)}
    vals = std_row.copy()
    mk = np.zeros_like(vals, dtype=bool)
    for s in kept_subjects:
        j = col_index.get(s)
        if j is not None: mk[j] = True
    vals[~mk] = np.nan
    miss = (~np.isfinite(vals)).astype(float)  # indicators
    vals = np.nan_to_num(vals, nan=0.0)
    return np.concatenate([vals, miss], axis=0)

def select_xgb_model_path(target: str, K: int, xgb_index: pd.DataFrame):
    df = xgb_index[xgb_index["target"] == target]
    if df.empty: return None
    if (df["K"] == K).any():
        return Path(df[df["K"] == K]["model_path"].iloc[0])
    # chọn K gần nhất nếu không có đúng K
    df2 = df.assign(diff=(df["K"] - K).abs()).sort_values("diff")
    return Path(df2.iloc[0]["model_path"])

def predict_mf_for_target(mf, means, stds, user_numeric: dict, target: str):
    if mf is None: return np.nan
    subjects = mf["subjects"]
    V = mf["V"]            # [n_items, k]
    b_i = mf["b_item"]     # [n_items]
    mu  = mf["mu"]         # ≈ 0
    lam = mf["lambda"]
    k   = mf["k"]
    col_index = {s:i for i,s in enumerate(subjects)}

    # standardize user vector
    std_vals = []
    for s in subjects:
        v = user_numeric.get(s, np.nan)
        if pd.isna(v):
            std_vals.append(np.nan)
        else:
            std_vals.append((float(v) - means[s]) / stds[s])
    std_vals = np.array(std_vals, dtype=float)

    obs_idx = np.where(np.isfinite(std_vals))[0]
    if obs_idx.size == 0:
        return np.nan

    V_K = V[obs_idx]                 # [n_obs, k]
    r   = std_vals[obs_idx]
    rhs = r - mu - b_i[obs_idx]
    A = V_K.T @ V_K + lam * np.eye(k)
    try:
        u_user = np.linalg.solve(A, V_K.T @ rhs)  # [k]
    except np.linalg.LinAlgError:
        u_user = np.linalg.pinv(A) @ (V_K.T @ rhs)

    t_idx = col_index.get(target, None)
    if t_idx is None:
        return np.nan
    y_std = mu + b_i[t_idx] + u_user @ V[t_idx]
    y = y_std * stds[target] + means[target]
    return float(y)

def predict_ggm_for_target(ggm, means, stds, user_numeric: dict, target: str, subjects: list):
    if ggm is None: return np.nan, None
    cov = ggm.get("cov", None)
    if cov is None: return np.nan, None

    idx = {s:i for i,s in enumerate(subjects)}

    # z-score user
    x = []
    O = []
    for s in subjects:
        v = user_numeric.get(s, np.nan)
        if pd.isna(v):
            x.append(np.nan)
        else:
            x.append((float(v) - means[s]) / stds[s])
            O.append(idx[s])

    if len(O) == 0 or target not in idx:
        return np.nan, None
    T = idx[target]
    O = np.array([o for o in O if o != T])
    if O.size == 0:
        return np.nan, None

    cov = np.asarray(cov)
    S_TO = cov[T, O].reshape(1, -1)
    S_OO = cov[np.ix_(O, O)]
    S_TT = cov[T, T]
    x_O  = np.array([x[o] for o in O])

    try:
        inv_S_OO = np.linalg.inv(S_OO)
    except np.linalg.LinAlgError:
        inv_S_OO = np.linalg.pinv(S_OO)

    y_std   = (S_TO @ inv_S_OO @ (x_O - 0.0)).item()   # mu=0 sau z-score
    var_T_O = float(S_TT - (S_TO @ inv_S_OO @ S_TO.T).item())
    y = y_std * stds[target] + means[target]
    return float(y), max(var_T_O, 1e-9)

# =========================
# Sidebar & Input (giữ nguyên UX)
# =========================
st.sidebar.header("1. Tải file điểm lên")
uploaded = st.sidebar.file_uploader("Chọn file Excel đầu vào theo mẫu input-score.xlsx", type=["xlsx", "xls"])

subjects, means, stds = load_subjects_means_stds()
st.sidebar.header("2. Chọn môn cần dự đoán")
target_subject = st.sidebar.selectbox("Môn học muốn dự đoán", subjects)

do_predict = st.sidebar.button("Dự đoán")

st.sidebar.markdown("---")
st.sidebar.subheader("📖 Hướng dẫn sử dụng")
st.sidebar.markdown(
    """
    1. **Tải file Excel**: Sử dụng mẫu `input-score.xlsx` để nhập danh sách môn học và điểm chữ đã đạt.
    2. **Chọn môn cần dự đoán** trong danh sách.
    3. Nhấn **Dự đoán** để xem kết quả.
    4. Nếu file thiếu dữ liệu nhiều môn, hệ thống sẽ báo lỗi.
    5. Điểm dự đoán hiển thị gồm **điểm chữ** và **điểm số chuẩn**.
    """
)

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

# =========================
# Chuẩn bị dữ liệu người dùng
# =========================
# map sang numeric GPA
user_numeric = {}
for _, row in df_raw.iterrows():
    subj = str(row["Môn học"]).strip()
    sc   = convert_letter_to_score(row["Điểm chữ"])
    if subj in subjects:
        user_numeric[subj] = sc

# Xác định K (số môn có điểm, trừ target)
kept_subjects = [s for s in subjects if (s != target_subject) and pd.notna(user_numeric.get(s, np.nan))]
K = len(kept_subjects)
if K < 5:
    st.error(f"Không đủ dữ liệu để dự đoán: chỉ có {K} môn hợp lệ, cần ít nhất 5 môn."); st.stop()

# build features cho XGB (chuẩn hoá + mask + indicators)
std_row = standardize_user_row(user_numeric, subjects, means, stds)
features = build_masked_features(std_row, kept_subjects, subjects).reshape(1, -1)

# =========================
# Hybrid predict: XGB theo (target,K) + MF + GGM
# =========================
xgb_index = load_xgb_index()
mf_art = load_mf()
ggm_art = load_ggm()

pred_xgb = np.nan
pred_mf  = np.nan
pred_ggm = np.nan
var_ggm  = None

# 1) XGB (target, K đúng/ gần nhất)
model_path = select_xgb_model_path(target_subject, K, xgb_index)
if model_path is not None and model_path.exists():
    try:
        xgb_model = joblib.load(model_path)
        y_std = float(xgb_model.predict(features)[0])  # model học trên thang standardized
        pred_xgb = y_std * stds[target_subject] + means[target_subject]
    except Exception:
        pred_xgb = np.nan

# 2) MF (luôn tính để blend nhẹ nếu có)
if mf_art is not None:
    pred_mf = predict_mf_for_target(mf_art, means, stds, user_numeric, target_subject)

# 3) GGM (conditional mean + variance)
if ggm_art is not None:
    pred_ggm, var_ggm = predict_ggm_for_target(ggm_art, means, stds, user_numeric, target_subject, subjects)

# 4) Blending (giữ output 1 số như UI cũ)
#    - Base: XGB ưu tiên (K<=10: 0.85, K>10: 0.70); MF = phần còn lại
#    - Nếu có GGM: cộng thêm tối đa 0.25 theo confidence = 1/(var+eps)
def do_blend(px, pm, pg, v_g, K):
    w_xgb = 0.85 if K <= 10 else 0.70
    w_mf  = 1.0 - w_xgb
    w_ggm = 0.0
    if np.isfinite(pg) and (v_g is not None) and (v_g > 0):
        conf = 1.0 / (v_g + 1e-6)
        frac = conf / (conf + 1.0)   # 0..1
        w_ggm = 0.25 * frac          # ≤ 0.25
        scale = 1.0 - w_ggm
        s = w_xgb + w_mf
        if s > 0:
            w_xgb *= scale / s
            w_mf  *= scale / s
    terms = []
    if np.isfinite(px): terms.append(px * w_xgb)
    if np.isfinite(pm): terms.append(pm * w_mf)
    if np.isfinite(pg): terms.append(pg * w_ggm)
    if not terms:
        return np.nan
    return float(np.sum(terms))

pred = do_blend(pred_xgb, pred_mf, pred_ggm, var_ggm, K)

if not np.isfinite(pred):
    st.error("Không thể dự đoán với dữ liệu hiện tại."); st.stop()

# =========================
# In kết quả (GIỮ Y HỆT CÁCH IN CỦA BẠN)
# =========================
try:
    st.subheader(f"🎯 Kết quả dự đoán cho môn **{target_subject}**")
except Exception:
    st.subheader("🎯 Kết quả dự đoán")

letter = numeric_to_letter(pred)
LETTER_TO_NUMERIC = {
    "A+": 4.0, "A": 4.0,
    "B+": 3.5, "B": 3.0,
    "C+": 2.5, "C": 2.0,
    "D+": 1.5, "D": 1.0,
}
base_letter = "A" if "A" in letter else letter  # xử lý "A / A+"
converted_numeric = LETTER_TO_NUMERIC.get(base_letter, np.nan)

st.success(f"- **Điểm chữ dự đoán:** {letter}")
st.success(f"- **Điểm số chuẩn tương ứng:** {converted_numeric:.1f}")
