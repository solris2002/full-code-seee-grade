# Ứng dụng học máy dự đoán kết quả học tập sinh viên

## Giới thiệu
Đồ án tập trung xây dựng ứng dụng học máy nhằm dự đoán kết quả học tập của sinh viên.  
Hệ thống hỗ trợ dự đoán:
- **GPA (Grade Point Average)** của học kỳ kế tiếp.  
- **CPA (Cumulative Point Average)** toàn khóa học.  
- **Điểm số môn học** thuộc các ngành/trường hợp nghiên cứu.  

Mục tiêu của hệ thống là giúp sinh viên, giảng viên và cố vấn học tập:
- Lên kế hoạch học tập hiệu quả.  
- Phát hiện sớm sinh viên có nguy cơ kết quả kém để có biện pháp hỗ trợ.  

## Công nghệ sử dụng
- **Ngôn ngữ**: Python  
- **Thư viện**: NumPy, Pandas, Scikit-learn, XGBoost, Joblib  
- **Mô hình chính**: Gaussian Graphical Model (GGM), Hồi quy, XGBoost  
- **Giao diện**: Streamlit, HTML/CSS  
- **Triển khai**: Local (Streamlit) hoặc các nền tảng như Streamlit Cloud, Hugging Face  

## Cài đặt
1. Clone repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Tạo môi trường ảo và cài đặt dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows

   pip install -r requirements.txt
   ```

## Cách chạy
Chạy ứng dụng giao diện web:
```bash
streamlit run app.py
```
Ứng dụng có thể triển khai trực tiếp trên Streamlit Cloud hoặc Hugging Face Spaces.

## Cấu trúc thư mục
```
├── Training/                      # Huấn luyện mô hình
│   ├── training-gpa-cpa/          # Huấn luyện GPA & CPA (hồi quy)
│   ├── training-find-score-et/    # Huấn luyện ET1 (nghiên cứu chính, nhiều thử nghiệm)
│   └── training-find-score-ee/    # Huấn luyện EE2 (kế thừa từ ET1)
│
├── Web/                           # Ứng dụng web
    ├── home/                      # Trang HTML tĩnh, giao diện chính
    ├── cpa-gpa/                   # Dự đoán CPA & GPA (Streamlit app)
    └── hust-grade-course/         # Dự đoán điểm môn học (Streamlit app)

```

## Tính năng chính
- **Dự đoán GPA/CPA:** hỗ trợ sinh viên nắm bắt tình hình học tập toàn khóa.  
- **Dự đoán điểm môn học:** cho phép sinh viên theo dõi kết quả từng môn cốt lõi.  
- **Giao diện trực quan:** triển khai bằng Streamlit, dễ sử dụng và mở rộng.  

## Định hướng phát triển
- Nghiên cứu các mô hình khác kết hợp hoặc thay thế Gaussian Graphical Model.  
- Tích hợp dữ liệu hành vi học tập, tham dự lớp, và tương tác để tăng độ chính xác.  
- Phát triển thêm chức năng khuyến nghị lộ trình học tập cá nhân hóa.  
