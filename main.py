import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="MLR Score Predictor", page_icon="", layout="centered")

st.title("MULTIPLE LINEAR REGRESSION PREDICTOR")
st.write("Mô hình hồi quy tuyến tính đa biến dự đoán điểm số.")
st.markdown("---")
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
    x1 = st.number_input("1. Số giờ tự học môn này trong tuần (giờ):", min_value=0, max_value=168, value=5)
    
    x2_vn = st.number_input("2. Điểm số của bài kiểm tra trước đó (bài gần nhất, 1 bài) (Thang điểm 10):", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    x2 = x2_vn * 10
    
    x3 = st.number_input("3. Số giờ ngủ trung bình mỗi ngày (giờ):", min_value=1, max_value=24, value=7)
    x4 = st.number_input("4. Số câu hỏi ôn tập liên quan đến đề thi đã tự giải:", min_value=0, max_value=500, value=30)

    st.markdown("---")

    if st.button("CHẠY MÔ HÌNH DỰ ĐOÁN"):
        input_data = np.array([[x1, x2, x3, x4]])
        prediction_100 = model.predict(input_data)
        prediction_100 = max(10.0, min(100.0, float(prediction_100[0])))
        final_score_vn = prediction_100 / 10

        st.success("📊 KẾT QUẢ PHÂN TÍCH TỪ MÔ HÌNH MLR:")
        st.metric(label="Điểm Số Dự Kiến", value=f"{final_score_vn:.1f} / 10")
        st.metric(label="Chỉ Số Năng Lực Gốc", value=f"{prediction_100:.1f} / 100")
        
        st.code(f"""
> Matrix computing initialized.
> Input Vector nạp vào ma trận: {input_data.tolist()}
> Status: Inference executed successfully.
        """)

except FileNotFoundError:
    st.error("⚠️ Không tìm thấy file 'model.pkl'! Bạn hãy chạy lệnh 'python train.py' ở Bước 1 trước nhé.")
