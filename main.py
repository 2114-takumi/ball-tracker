import cv2
import time
import numpy as np

# カメラの起動
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラを起動できませんでした")
    exit()

# FPS測定用
start_time = time.perf_counter()
frame_count = 0

# モルフォロジー処理用カーネル
kernel = np.ones((9, 9), np.uint8)

# ノイズ除去のための最小輪郭面積
min_area = 500

# 検出するボールの半径の範囲（ピクセル）
min_radius = 10
max_radius = 100

trajectory = []
previous_center = None
previous_time = None

# 比較する移動平均の窓幅
window_sizes = [1, 5, 10]

# 速度履歴は最大10回分保存
smoothing_window = max(window_sizes)

# 画面表示にはWindow=5を使用
display_window = 5

vx_history = []
vy_history = []

# 予測時間
prediction_time = 0.2

# 各Windowの予測・誤差を別々に管理
pending_predictions = {w: [] for w in window_sizes}
prediction_errors = {w: [] for w in window_sizes}
all_prediction_errors = {w: [] for w in window_sizes}


while True:

    # ① カメラ画像を取得
    ret, frame = cap.read()

    if not ret:
        print("画像を取得できませんでした")
        break

    frame_count += 1

    # ② BGRからHSVへ変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ③ 白色領域を抽出
    mask = cv2.inRange(
        hsv,
        (0, 0, 170),
        (179, 75, 255)
    )

    # ④ クロージングで穴埋め
    cleaned_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ⑤ 輪郭検出
    contours, hierarchy = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 検出結果の初期化
    area = 0
    ball_center = None
    ball_radius = None

    # ⑥ 最大輪郭を取得
    if contours:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(largest_contour)

        # 小さなノイズを除外
        if area >= min_area:

            # ⑦ ボールの中心座標と半径を取得
            center, radius = cv2.minEnclosingCircle(
                largest_contour
            )

            if min_radius <= radius <= max_radius:
                x, y = center

                x = int(x)
                y = int(y)
                trajectory.append((x, y))
                if len(trajectory) > 30:
                    trajectory.pop(0)
                radius = int(radius)

                ball_center = (x, y)
                ball_radius = radius

                # ⑧ 緑色の外接円
                cv2.circle(
                    frame,
                    (x, y),
                    radius,
                    (0, 255, 0),
                    2
                )

                # ⑨ 赤色の中心点
                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 0, 255),
                    -1
                )

                # ⑩ 中心座標を表示
                cv2.putText(
                    frame,
                    f"({x}, {y})",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

    # 速度計算
    vx = None
    vy = None
    speed = None
    avg_vx = None
    avg_vy = None
    
    if ball_center is not None:
        detection_time = time.perf_counter()
        if previous_center is not None and previous_time is not None:
            dt = detection_time - previous_time
            if dt > 0:
                vx = (ball_center[0] - previous_center[0]) / dt
                vy = (ball_center[1] - previous_center[1]) / dt

                speed = np.hypot(vx, vy)

                vx_history.append(vx)
                vy_history.append(vy)

                # 最大10回分の速度を保存
                if len(vx_history) > smoothing_window:
                    vx_history.pop(0)
                    vy_history.pop(0)

                # 画面表示用：Window=5の平均速度
                avg_vx = np.mean(vx_history[-display_window:])
                avg_vy = np.mean(vy_history[-display_window:])

                # 10回分の速度がそろってから公平に比較
                if len(vx_history) == smoothing_window:
                    for window in window_sizes:
                        mean_vx = np.mean(vx_history[-window:])
                        mean_vy = np.mean(vy_history[-window:])

                        pred_x = ball_center[0] + mean_vx * prediction_time
                        pred_y = ball_center[1] + mean_vy * prediction_time
                        target_time = detection_time + prediction_time

                        pending_predictions[window].append(
                            (target_time, (pred_x, pred_y))
                        )

        previous_center = ball_center
        previous_time = detection_time
    else:
        previous_center = None
        previous_time = None

        trajectory.clear()
        vx_history.clear()
        vy_history.clear()
        for window in window_sizes:
            pending_predictions[window].clear()


    # 速度ベクトルを矢印で表示
    if ball_center is not None and avg_vx is not None and avg_vy is not None:

        # 矢印の始点
        start_point = ball_center

        # 矢印の長さを調整する時間
        arrow_time = 0.1

        # 矢印の終点
        x_end = int(ball_center[0] + avg_vx * arrow_time)
        y_end = int(ball_center[1] + avg_vy * arrow_time)

        end_point = (x_end, y_end)

        # 矢印を描画
        cv2.arrowedLine(
            frame,
            start_point,
            end_point,
            (255, 255, 0),
            3,
            tipLength=0.2
        ) 

        prediction_time = 0.2

        predicted_x = int(ball_center[0] + avg_vx * prediction_time)                
        predicted_y = int(ball_center[1] + avg_vy * prediction_time)

        cv2.circle(
            frame,
            (predicted_x, predicted_y),
            6,
            (255, 0, 255),
            -1
        )

    # 3種類の予測誤差を同時に評価
    if ball_center is not None:
        for window in window_sizes:
            while (
                pending_predictions[window]
                and pending_predictions[window][0][0] <= detection_time
            ):
                target_time, predicted_center = (
                    pending_predictions[window].pop(0)
                )
                error = np.hypot(
                    ball_center[0] - predicted_center[0],
                    ball_center[1] - predicted_center[1]
                )
                prediction_errors[window].append(error)
                all_prediction_errors[window].append(error)

    for i in range(1, len(trajectory)):
        point1 = trajectory[i - 1]
        point2 = trajectory[i]
        cv2.line(frame, point1, point2, (255, 0, 0), 2)

    # ⑪ FPS測定
    current_time = time.perf_counter()
    elapsed_time = current_time - start_time


    if elapsed_time >= 1.0:

        fps = frame_count / elapsed_time

        print(f"FPS: {fps:.2f}")
        print(f"Largest Contour Area: {area:.1f}")
        print(f"Trajectory points: {len(trajectory)}")
        if speed is not None:
            print(f"Speed: {speed:.2f} px/s")

        if ball_center is not None:
            print(f"Ball Center: {ball_center}")
            print(f"Ball Radius: {ball_radius}")
        else:
            print("Ball not detected")

        if vx is not None:
            print(f"Velocity X: {vx:.2f} px/s")

        if vy is not None:
            print(f"Velocity Y: {vy:.2f} px/s") 

        if vx is not None and vy is not None and avg_vx is not None and avg_vy is not None:
            print(f"Raw vx: {vx:.2f} px/s")
            print(f"Average vx: {avg_vx:.2f} px/s")
            print(f"Raw vy: {vy:.2f} px/s")
            print(f"Average vy: {avg_vy:.2f} px/s")

        for window in window_sizes:
            if prediction_errors[window]:
                mean_error = np.mean(prediction_errors[window])
                print(
                    f"Window={window}: "
                    f"Mean Prediction Error: {mean_error:.2f} px"
                )
                prediction_errors[window].clear()
            
        start_time = current_time
        frame_count = 0


    # ⑫ 映像を表示
    cv2.imshow("White Mask", mask)
    cv2.imshow("Cleaned Mask", cleaned_mask)
    cv2.imshow("Ball Detection", frame)
    # qキーで終了
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

# ⑬ 終了処理
print("\n=== Prediction Comparison ===")
for window in window_sizes:
    errors = all_prediction_errors[window]
    if errors:
        overall_mean_error = np.mean(errors)
        print(
            f"Window={window}: "
            f"Total Predictions={len(errors)}, "
            f"Overall Mean Error={overall_mean_error:.2f} px"
        )

cap.release()
cv2.destroyAllWindows()