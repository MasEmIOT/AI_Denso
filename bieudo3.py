import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
from datetime import datetime, timedelta
from collections import deque

# Khởi tạo ứng dụng Dash
app = dash.Dash(__name__)

# Khởi tạo dữ liệu thực tế với mỗi giây một giá trị
current_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
time_actual = deque([current_time + timedelta(seconds=i) for i in range(60)], maxlen=120)  # Dữ liệu thực từ 0s đến 60s
values_actual = deque(np.sin(0.1 * np.arange(60)) + np.random.normal(0, 0.05, 60), maxlen=120)

# Dữ liệu dự đoán ban đầu
time_predicted_old = [current_time + timedelta(seconds=60 + i) for i in range(20)]  # Dự đoán từ 60s đến 80s
values_predicted_old = np.sin(0.1 * np.arange(60, 80)) + np.random.normal(0, 0.05, 20)

# Dữ liệu dự đoán mới
time_predicted_new = time_predicted_old
values_predicted_new = values_predicted_old
threshold = 0.82  # Ngưỡng cảnh báo

# Tạo bố cục của ứng dụng
app.layout = html.Div([
    html.H1("Biểu đồ Dự báo Độ rung Thời gian Thực", style={'text-align': 'center'}),
    dcc.Graph(id='live-update-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # Cập nhật mỗi giây
        n_intervals=0
    )
])

# Biến toàn cục để lưu lại các đường dự đoán cũ
comparison_predictions = deque(maxlen=5)

# Hàm cập nhật dữ liệu
@app.callback(
    Output('live-update-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph_live(n):
    global time_actual, values_actual, time_predicted_old, values_predicted_old, time_predicted_new, values_predicted_new, comparison_predictions

    # Cập nhật mỗi 60 giây để tạo dự đoán mới
    if n % 60 == 0 and n > 0:
        # Thêm 20 giây đầu của dự đoán cũ vào dữ liệu thực
        time_actual.extend(time_predicted_old[:20])
        values_actual.extend(values_predicted_old[:20])

        # Lưu lại 20 giây đầu của đường dự đoán cũ
        comparison_predictions.append((time_predicted_old[:20], values_predicted_old[:20]))

        # Tạo đường dự đoán mới cho 20 giây tiếp theo
        last_time_actual = time_actual[-1]
        time_predicted_new = [last_time_actual + timedelta(seconds=i) for i in range(1, 21)]
        values_predicted_new = (np.sin(0.1 * np.arange(len(time_actual), len(time_actual) + 20)) +
                                np.random.normal(0, 0.05, 20)) * 0.9  # Thêm nhiễu nhẹ vào dự đoán

        # Cập nhật đường dự đoán cũ
        time_predicted_old = time_predicted_new
        values_predicted_old = values_predicted_new

    # Cập nhật trục thời gian (60 giây gần nhất)
    x_start_time = time_actual[-60]
    x_end_time = time_predicted_new[-1]

    # Tạo biểu đồ
    fig = go.Figure()

    # Dữ liệu thực (nối đường + điểm chấm nhỏ mỗi giây)
    fig.add_trace(go.Scatter(
        x=list(time_actual),
        y=list(values_actual),
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue'),
        marker=dict(color='blue', size=4)
    ))

    # Hiển thị các đường "Previous Predict" với nét đứt màu đỏ đậm
    for i, (time_pred, values_pred) in enumerate(comparison_predictions):
        fig.add_trace(go.Scatter(
            x=list(time_pred),
            y=list(values_pred),
            mode='lines',
            name=f'Previous Predict {i+1}' if i == 0 else None,
            line=dict(color='red', dash='dash'),
            showlegend=i == 0
        ))

    # Đường dự đoán mới với nét đứt màu cam
    fig.add_trace(go.Scatter(
        x=list(time_predicted_new),
        y=list(values_predicted_new),
        mode='lines',
        name='New Predict',
        line=dict(color='orange', dash='dot')
    ))

    # Vùng màu nền dự đoán
    fig.add_vrect(
        x0=time_predicted_new[0], x1=time_predicted_new[-1],
        fillcolor="green", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Vùng dự đoán", annotation_position="top left"
    )

    # Đường ngưỡng cảnh báo
    fig.add_hline(y=threshold, line=dict(color='red', dash='dot'), name="Ngưỡng cảnh báo")

    # Định dạng trục X
    fig.update_layout(
        title="Dự báo xu hướng độ rung thời gian thực",
        xaxis=dict(
            title="Thời gian (HH:MM:SS)",
            tickformat="%H:%M:%S",
            range=[x_start_time, x_end_time]
        ),
        yaxis=dict(title="Giá trị độ rung"),
        showlegend=True
    )

    return fig

# Chạy ứng dụng
if __name__ == '__main__':
    app.run_server(debug=True)
