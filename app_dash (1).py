import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from data_test import get_next_data_point, predict_next_20
from collections import deque
from datetime import timedelta

# Tạo ứng dụng Dash
app = dash.Dash(__name__)

# Biến toàn cục
time_actual = deque([], maxlen=100)
values_actual_x = deque([], maxlen=100)
values_actual_y = deque([], maxlen=100)
values_actual_z = deque([], maxlen=100)

comparison_predictions_x = []  # Lưu trữ các đường dự đoán trục X
comparison_predictions_y = []  # Lưu trữ các đường dự đoán trục Y
comparison_predictions_z = []  # Lưu trữ các đường dự đoán trục Z

# Giao diện ứng dụng
app.layout = html.Div([
    html.H1("Biểu đồ Dự báo Độ rung Thời gian Thực", style={'text-align': 'center'}),
    dcc.Graph(id='live-update-graph-x'),
    dcc.Graph(id='live-update-graph-y'),
    dcc.Graph(id='live-update-graph-z'),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # Cập nhật mỗi giây
        n_intervals=0
    )
])

# Callback cập nhật biểu đồ
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

    # Lấy dữ liệu tiếp theo
    next_time, next_values = get_next_data_point()
    if next_time:
        time_actual.append(next_time)
        values_actual_x.append(next_values[0])
        values_actual_y.append(next_values[1])
        values_actual_z.append(next_values[2])

    # Khi đủ 80 điểm, thực hiện dự đoán
    if len(values_actual_x) == 80:
        data_window = list(zip(values_actual_x, values_actual_y, values_actual_z))
        predictions = predict_next_20(data_window)
        time_predicted = [time_actual[-1] + timedelta(seconds=i + 1) for i in range(20)]

        # Lưu lại các đường dự đoán
        comparison_predictions_x.append((time_predicted, predictions[:, 0]))
        comparison_predictions_y.append((time_predicted, predictions[:, 1]))
        comparison_predictions_z.append((time_predicted, predictions[:, 2]))

    # Tạo biểu đồ cho từng trục
    fig_x = create_figure(time_actual, values_actual_x, comparison_predictions_x, "Trục X")
    fig_y = create_figure(time_actual, values_actual_y, comparison_predictions_y, "Trục Y")
    fig_z = create_figure(time_actual, values_actual_z, comparison_predictions_z, "Trục Z")

    return fig_x, fig_y, fig_z

# Hàm tạo biểu đồ
def create_figure(time_actual, values_actual, comparison_predictions, title):
    fig = go.Figure()

    # Giá trị thực
    fig.add_trace(go.Scatter(
        x=list(time_actual),
        y=list(values_actual),
        mode='lines+markers',
        name='Giá trị thực',
        line=dict(color='blue'),
        marker=dict(size=6)
    ))

    # Đường dự đoán cũ
    for idx, (time_pred, values_pred) in enumerate(comparison_predictions[:-1]):
        fig.add_trace(go.Scatter(
            x=list(time_pred),
            y=list(values_pred),
            mode='lines',
            name=f'Dự đoán trước {idx + 1}',
            line=dict(color='red', dash='dash')
        ))

    # Đường dự đoán mới nhất
    if comparison_predictions:
        time_pred_new, values_pred_new = comparison_predictions[-1]
        fig.add_trace(go.Scatter(
            x=list(time_pred_new),
            y=list(values_pred_new),
            mode='lines',
            name='Dự đoán mới',
            line=dict(color='orange', dash='dot')
        ))

        # Vùng dự đoán
        fig.add_vrect(
            x0=time_pred_new[0],
            x1=time_pred_new[-1],
            fillcolor="green",
            opacity=0.1,
            layer="below",
            line_width=0,
            annotation_text="Vùng dự đoán",
            annotation_position="top left"
        )

    # Ngưỡng
    fig.add_hline(y=0.8, line_dash="dot", line_color="green", annotation_text="Ngưỡng trên")
    fig.add_hline(y=0.2, line_dash="dot", line_color="red", annotation_text="Ngưỡng dưới")

    # Cập nhật bố cục
    fig.update_layout(
        title=title,
        xaxis=dict(title="Thời gian"),
        yaxis=dict(title="Giá trị độ rung"),
        showlegend=True
    )
    return fig

# Chạy ứng dụng
if __name__ == '__main__':
    app.run_server(debug=True)
