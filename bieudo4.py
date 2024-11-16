import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
from datetime import datetime, timedelta

# Tải mô hình LSTM đã huấn luyện
model = load_model('lstm_model.h5')

# Đọc dữ liệu từ file CSV
data = pd.read_csv('vibration_data_with_timestamp.csv')

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data[['x', 'y', 'z']])

# Lấy cột đầu tiên (giả sử bạn chỉ dự đoán theo một trục)
values = data_normalized[:, 0]

# Khởi tạo dữ liệu thực tế với mỗi giây một giá trị
current_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
time_actual = deque([current_time + timedelta(seconds=i) for i in range(80)], maxlen=120)
values_actual = deque(values[:80], maxlen=120)

# Dữ liệu dự đoán ban đầu
time_predicted_old = []
values_predicted_old = []

# Biến để lưu các đường dự đoán cũ
comparison_predictions = deque(maxlen=5)

# Khởi tạo ứng dụng Dash
app = dash.Dash(__name__)

# Giao diện của ứng dụng
app.layout = html.Div([
    html.H1("Biểu đồ Dự báo Độ rung Thời gian Thực", style={'text-align': 'center'}),
    dcc.Graph(id='live-update-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # Cập nhật mỗi giây
        n_intervals=0
    )
])

# Hàm cập nhật biểu đồ
@app.callback(
    Output('live-update-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph_live(n):
    global time_actual, values_actual, time_predicted_old, values_predicted_old, comparison_predictions

    # Cập nhật dữ liệu thực tế
    if n < len(values):
        time_actual.append(time_actual[-1] + timedelta(seconds=1))
        values_actual.append(values[n])

    # Khi có đủ 80 giá trị, bắt đầu dự đoán
    if len(values_actual) >= 80 and n % 20 == 0 and n > 0:
        # Chuẩn bị đầu vào cho mô hình
        input_data = np.array(values_actual)[-80:].reshape(1, 80, 1)
        prediction = model.predict(input_data)[0]

        # Cập nhật đường dự đoán
        last_time = time_actual[-1]
        time_predicted_new = [last_time + timedelta(seconds=i) for i in range(1, 21)]
        values_predicted_new = scaler.inverse_transform([[p, 0, 0] for p in prediction])[:, 0]

        # Lưu đường dự đoán cũ
        comparison_predictions.append((time_predicted_old, values_predicted_old))

        # Cập nhật đường dự đoán mới
        time_predicted_old = time_predicted_new
        values_predicted_old = values_predicted_new

    # Tạo biểu đồ
    fig = go.Figure()

    # Thêm dữ liệu thực tế
    fig.add_trace(go.Scatter(
        x=list(time_actual),
        y=list(values_actual),
        mode='lines+markers',
        name='Giá trị thực',
        line=dict(color='blue'),
        marker=dict(size=4, color='blue')
    ))

    # Thêm các đường dự đoán cũ
    for i, (time_pred, values_pred) in enumerate(comparison_predictions):
        fig.add_trace(go.Scatter(
            x=time_pred,
            y=values_pred,
            mode='lines',
            name=f'Previous Predict {i+1}',
            line=dict(color='red', dash='dash'),
            showlegend=(i == 0)
        ))

    # Thêm đường dự đoán mới
    if time_predicted_old and values_predicted_old:
        fig.add_trace(go.Scatter(
            x=time_predicted_old,
            y=values_predicted_old,
            mode='lines',
            name='New Predict',
            line=dict(color='orange', dash='dot')
        ))

    # Vùng dự đoán
    if time_predicted_old:
        fig.add_vrect(
            x0=time_predicted_old[0], x1=time_predicted_old[-1],
            fillcolor="green", opacity=0.1,
            layer="below", line_width=0
        )

    # Định dạng biểu đồ
    fig.update_layout(
        title="Dự báo xu hướng độ rung thời gian thực",
        xaxis=dict(title="Thời gian (HH:MM:SS)", tickformat="%H:%M:%S"),
        yaxis=dict(title="Giá trị độ rung"),
        showlegend=True
    )

    return fig

# Chạy ứng dụng
if __name__ == '__main__':
    app.run_server(debug=True)
