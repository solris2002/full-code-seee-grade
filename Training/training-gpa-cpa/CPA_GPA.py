import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Dự đoán kết quả học tập", layout="centered")
st.title(":blue[DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN]")

# =============================
# 1. Sidebar chọn loại sinh viên
# =============================
st.sidebar.subheader("Cài đặt đầu vào")


# Chọn loại sinh viên: Cử nhân (8 kỳ) hoặc Kỹ sư (10 kỳ)
student_type = st.sidebar.selectbox("Định hướng sinh viên", ("Cử nhân", "Kỹ sư"))
max_semester = 6 if student_type == "Cử nhân" else 8

# Chọn kỳ hiện tại
current_semester = st.sidebar.selectbox("Kỳ hiện tại", list(range(1, max_semester + 1)))

# Nhập GPA từ kỳ 1 đến kỳ hiện tại
data_inputs = []
gpa_inputs = []
tc_inputs = []

for i in range(1, current_semester + 1):
    gpa = st.sidebar.number_input(f"GPA kỳ {i}", min_value=0.0, max_value=4.0, step=0.01, format="%.2f")
    gpa_inputs.append(gpa)
    data_inputs.append(gpa)
    tc_qua = st.sidebar.number_input(f"Tín chỉ kỳ {i}", min_value=0, format="%d")
    data_inputs.append(tc_qua)
    tc_inputs.append(tc_qua)


# =============================
# 2. Xử lý đầu vào và dự đoán
# =============================
if any(g == 0.0 for g in data_inputs) or any(t < 0 for t in data_inputs):
    st.warning("⚠️ Vui lòng nhập đầy đủ GPA  cho tất cả các kỳ đã chọn.")
else:
    try:
        input_data = np.array(data_inputs).reshape(1, -1)
        model_prefix = "8" if student_type == "Cử nhân" else "10"

        # Định danh nhóm
        group_key_cpa = f"GPA_TC_1_{current_semester}" if current_semester > 1 else "GPA_TC_1"

        # =============================
        # 3. Dự đoán Final CPA
        # =============================
        cpa_model_path = f"models_streamlit/final_cpa_{model_prefix}_ki.joblib"
        cpa_dict = joblib.load(cpa_model_path)

        model_cpa = cpa_dict[group_key_cpa]

        predicted_cpa = model_cpa.predict(input_data)[0]

        st.subheader("🎓 Dự đoán CPA tốt nghiệp:")
        st.success(f"CPA Tốt Nghiệp: {predicted_cpa:.2f}")

        # =============================
        # 4. Dự đoán GPA kỳ tiếp theo
        # =============================
        if current_semester < max_semester:
            group_key_gpa = f"GPA_{current_semester + 1}" 
            next_gpa_path = f"models_streamlit/next_gpa_{model_prefix}_ki.joblib"
            next_dict = joblib.load(next_gpa_path)

            # scaler_next = next_dict[group_key]['scaler']
            model_next = next_dict[group_key_gpa]

            # input_scaled_next = scaler_next.transform(input_data)
            predicted_next_gpa = model_next.predict(input_data)[0]

            st.subheader(f"📘 Dự đoán GPA kỳ {current_semester + 1}:")
            st.info(f"GPA dự đoán: {predicted_next_gpa:.2f}")

    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi khi dự đoán: {e}")
