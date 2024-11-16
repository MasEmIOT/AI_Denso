import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from collections import deque

# Tải mô hình LSTM đã huấn luyện
model = load_model('lstm_model.h5')

# Chuẩn hóa dữ liệu
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_z = MinMaxScaler(feature_range=(0, 1))

# Đọc dữ liệu từ CSV
file_path = 'vibration_data_with_timestamp.csv'
df = pd.read_csv(file_path)

# Làm sạch dữ liệu
df['time'] = range(len(df))  # Chuyển thời gian thành số giây từ 0, 1, 2,...
data_x = df['x'].values.reshape(-1, 1)  # Giá trị trục X
data_y = df['y'].values.reshape(-1, 1)  # Giá trị trục Y
data_z = df['z'].values.reshape(-1, 1)  # Giá trị trục Z

# Chuẩn hóa dữ liệu
data_x = scaler_x.fit_transform(data_x)
data_y = scaler_y.fit_transform(data_y)
data_z = scaler_z.fit_transform(data_z)

# Biến toàn cục
current_time_index = 0  # Chỉ mục hiện tại để theo dõi dòng dữ liệu
timestamps = df['time'].values  # Thời gian tương ứng (tính bằng giây)
values_actual_x = deque([], maxlen=100)  # Hàng đợi chứa giá trị trục X
values_actual_y = deque([], maxlen=100)  # Hàng đợi chứa giá trị trục Y
values_actual_z = deque([], maxlen=100)  # Hàng đợi chứa giá trị trục Z

# Hàm lấy điểm dữ liệu tiếp theo
def get_next_data_point():
    global current_time_index
    if current_time_index < len(data_x):
        next_time = timestamps[current_time_index]
        next_x = data_x[current_time_index][0]
        next_y = data_y[current_time_index][0]
        next_z = data_z[current_time_index][0]
        current_time_index += 1
        return next_time, next_x, next_y, next_z
    else:
        return None, None, None, None  # Hết dữ liệu

# Hàm dự đoán
def predict_next_20(data, scaler):
    input_data = np.array(data).reshape(1, 80, 1)
    prediction = model.predict(input_data)[0]
    return scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()  # Đưa dữ liệu về dạng gốc
