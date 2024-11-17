import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Tải mô hình đã huấn luyện
model = load_model("lstm_model.h5")

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))

# Đọc dữ liệu từ file CSV
file_path = "M01_Feb_2019_OP12_000_bad.csv"
df = pd.read_csv(file_path)

# Kiểm tra dữ liệus
print("Cột trong file CSV:", df.columns)

# Nếu không có cột 'time', tạo thời gian giả lập
if 'time' not in df.columns:
    print("Cột 'time' không tồn tại. Thêm thời gian giả lập.")
    df['time'] = pd.date_range(start="2024-01-01", periods=len(df), freq="S")  # Tạo cột thời gian mỗi giây

# Làm sạch dữ liệu và chuẩn hóa
df.fillna(method='ffill', inplace=True)
data = scaler.fit_transform(df[['x', 'y', 'z']].values)

# Biến toàn cục
current_time_index = 0  # Chỉ mục để đọc dữ liệu
timestamps = pd.to_datetime(df['time'], errors='coerce').fillna(method='ffill')  # Đọc cột thời gian

# Hàm lấy dữ liệu tiếp theo
def get_next_data_point():
    global current_time_index
    if current_time_index < len(data):
        next_time = timestamps[current_time_index]
        next_values = data[current_time_index]
        current_time_index += 1
        return next_time, next_values
    return None, None

# Hàm dự đoán
def predict_next_20(input_data):
    input_data = np.array(input_data).reshape(1, 80, 3)
    predictions = model.predict(input_data)
    return scaler.inverse_transform(predictions[0])  # Đưa dự đoán về giá trị gốc

# Test các hàm
print("Lấy điểm dữ liệu tiếp theo:", get_next_data_point())
