import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from data_test import (
    get_next_data_point,
    values_actual_x, values_actual_y, values_actual_z,
    predict_next_20, scaler_x, scaler_y, scaler_z
)
from collections import deque

# Tạo ứng dụng Dash
app = dash.Dash(__name__)

# Biến toàn cục
time_actual = deque([], maxlen=100)  # Dữ liệu thời gian
comparison_predictions_x = deque(maxlen=5)  # Dự đoán trục X
comparison_predictions_y = deque(maxlen=5)  # Dự đoán trục Y
comparison_predictions_z = deque(maxlen=5)  # Dự đoán trục Z

# Giao diện của ứng dụng
app.layout = html.Div([
    html.H1("Biểu đồ Dự báo Độ rung Thời gian Thực", style={'text-align': 'center'}),
    
    # Biểu đồ trục X
    dcc.Graph(id='live-update-graph-x'),
    
    # Biểu đồ trục Y
    dcc.Graph(id='live-update-graph-y'),
    
    # Biểu đồ trục Z
    dcc.Graph(id='live-update-graph-z'),
    
    # Cập nhật mỗi giây
    dcc.Interval(
        id='interval-component',
        interval=1000,  # Cập nhật mỗi giây
        n_intervals=0
    )
])

# Callback để cập nhật biểu đồ
@app.callback(
    [
        Output('live-update-graph-x', 'figure'),
        Output('live-update-graph-y', 'figure'),
        Output('live-update-graph-z', 'figure')
    ],
    Input('interval-component', 'n_intervals')
)
def update_graph_live(n):
    global time_actual, values_actual_x, values_actual_y, values_actual_z
    global comparison_predictions_x, comparison_predictions_y, comparison_predictions_z

    # Lấy dữ liệu tiếp theo từ CSV
    next_time, next_x, next_y, next_z = get_next_data_point()
    if next_time is not None:
        time_actual.append(next_time)
        values_actual_x.append(next_x)
        values_actual_y.append(next_y)
        values_actual_z.append(next_z)

    # Khi đã đủ 80 giá trị, dự đoán
    if len(values_actual_x) == 80:
        # Dự đoán cho trục X
        prediction_x = predict_next_20(list(values_actual_x), scaler_x)
        time_predicted_x = [time_actual[-1] + i + 1 for i in range(20)]
        comparison_predictions_x.append((time_predicted_x, prediction_x))

        # Dự đoán cho trục Y
        prediction_y = predict_next_20(list(values_actual_y), scaler_y)
        comparison_predictions_y.append((time_predicted_x, prediction_y))

        # Dự đoán cho trục Z
        prediction_z = predict_next_20(list(values_actual_z), scaler_z)
        comparison_predictions_z.append((time_predicted_x, prediction_z))

    # Tạo biểu đồ cho từng trục
    fig_x = create_figure(time_actual, values_actual_x, comparison_predictions_x, "Trục X")
    fig_y = create_figure(time_actual, values_actual_y, comparison_predictions_y, "Trục Y")
    fig_z = create_figure(time_actual, values_actual_z, comparison_predictions_z, "Trục Z")

    return fig_x, fig_y, fig_z

# Hàm tạo biểu đồ
def create_figure(time_actual, values_actual, comparison_predictions, title):
    fig = go.Figure()

    # Hiển thị dữ liệu thực tế
    fig.add_trace(go.Scatter(
        x=list(time_actual),
        y=list(values_actual),
        mode='lines+markers',
        name='Giá trị thực',
        line=dict(color='blue'),
        marker=dict(color='blue', size=4)
    ))

    # Hiển thị các đường dự đoán
    for i, (time_pred, values_pred) in enumerate(comparison_predictions):
        fig.add_trace(go.Scatter(
            x=list(time_pred),
            y=list(values_pred),
            mode='lines',
            name=f'Dự đoán {i+1}',
            line=dict(color='orange', dash='dash'),
            showlegend=(i == 0)  # Chỉ hiển thị chú thích cho dự đoán đầu tiên
        ))

    # Định dạng biểu đồ
    fig.update_layout(
        title=title,
        xaxis=dict(title="Thời gian (giây)"),
        yaxis=dict(title="Giá trị độ rung"),
        showlegend=True
    )
    return fig

# Chạy ứng dụng
if __name__ == '__main__':
    app.run_server(debug=True)
