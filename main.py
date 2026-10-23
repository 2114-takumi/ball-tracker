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

# STEP 38-3：検出率測定用
total_frames = 0
detected_frames = 0

# モルフォロジー処理用カーネル
kernel = np.ones((9, 9), np.uint8)

# ノイズ除去のための最小輪郭面積
min_area = 500

# ボールとして認識する半径の範囲
min_radius = 50
max_radius = 100

# 円形度の最小値
min_circularity = 0.70

trajectory = []
previous_center = None
previous_time = None

# 前回追跡したボールの中心
tracking_center = None

# 前回位置から許容する最大移動距離
max_tracking_distance = 50

# 検出失敗の回数
missed_frames = 0
max_missed_frames = 5

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
motion_states = ["Stationary", "Moving", "Turning"]
state_prediction_errors = {
    state: {w: [] for w in window_sizes}
    for state in motion_states
}

# STEP 37：適応的Window選択の評価
adaptive_errors = []
adaptive_errors_recent = []

# どのWindowが何回選ばれたか
adaptive_window_counts = {1: 0, 5: 0}


while True:

    # ① カメラ画像を取得
    ret, frame = cap.read()

    if not ret:
        print("画像を取得できませんでした")
        break

    frame_count += 1
    total_frames += 1

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

    # STEP 38-1：円形度によるボール候補の選別
    ball_candidates = []
    ball_circularity = None

    # 最大輪郭の面積（既存のログ表示用）
    area = max(
        (cv2.contourArea(c) for c in contours),
        default=0
    )

    for contour in contours:
        # 輪郭面積
        contour_area = cv2.contourArea(contour)
        if contour_area < min_area:
            continue

        # 輪郭の周囲長
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        # 円形度を計算
        circularity = (
            4 * np.pi * contour_area / (perimeter ** 2)
        )
        if circularity < min_circularity:
            continue

        # 外接円の中心と半径
        center, radius = cv2.minEnclosingCircle(contour)
        if not (min_radius <= radius <= max_radius):
            continue

        ball_candidates.append((contour_area, center, radius, circularity))

    selected = None
    if ball_candidates:
        # 前回位置がない場合は、最大面積の候補を選択
        if tracking_center is None:
            selected = max(
                ball_candidates,
                key=lambda item: item[0]
            )
        else:
            # 前回位置に最も近い候補を探す
            closest = min(
                ball_candidates,
                key=lambda item: np.hypot(
                    item[1][0] - tracking_center[0],
                    item[1][1] - tracking_center[1]
                )
            )
            distance = np.hypot(
                closest[1][0] - tracking_center[0],
                closest[1][1] - tracking_center[1]
            )

            # 距離が50px以内なら採用
            if distance <= max_tracking_distance:
                selected = closest

    if selected is not None:
        selected_area, center, radius, ball_circularity = selected
        x, y = center

        x = int(x)
        y = int(y)
        trajectory.append((x, y))
        if len(trajectory) > 30:
            trajectory.pop(0)
        radius = int(radius)

        ball_center = (x, y)
        tracking_center = ball_center
        missed_frames = 0
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

    else:
        missed_frames += 1
        if missed_frames >= max_missed_frames:
            tracking_center = None

    # STEP 36-1：運動状態の判定
    motion_state = "Measuring"
    if ball_center is None:
        motion_state = "Not detected"
    elif len(trajectory) >= 6:
        # 直近5フレーム間の位置変化
        dx_recent = trajectory[-1][0] - trajectory[-6][0]
        dy_recent = trajectory[-1][1] - trajectory[-6][1]

        # 移動距離を計算
        distance = np.hypot(dx_recent, dy_recent)

        if distance < 4:
            motion_state = "Stationary"
        else:
            motion_state = "Moving"

            # 直近とその前の5フレームの移動方向を比較
            if len(trajectory) >= 11:
                dx_before = trajectory[-6][0] - trajectory[-11][0]
                if (
                    abs(dx_recent) >= 4
                    and abs(dx_before) >= 4
                    and dx_recent * dx_before < 0
                ):
                    motion_state = "Turning"

    if motion_state == "Turning":
        print("Turning detected!")

    # ボールを検出できたフレームを数える
    if ball_center is not None:
        detected_frames += 1

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
                            (target_time, (pred_x, pred_y), motion_state)
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


    # 運動状態を画面左上に表示
    cv2.putText(
        frame,
        f"Motion: {motion_state}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # STEP 37-2：表示するWindowを自動選択
    if (
        ball_center is not None
        and vx is not None
        and vy is not None
        and avg_vx is not None
        and avg_vy is not None
    ):
        if motion_state == "Turning":
            selected_window = 1
            display_vx = vx
            display_vy = vy
        else:
            selected_window = 5
            display_vx = avg_vx
            display_vy = avg_vy

        # 矢印の始点
        start_point = ball_center

        # 矢印の長さを調整する時間
        arrow_time = 0.1

        # 矢印の終点
        x_end = int(ball_center[0] + display_vx * arrow_time)
        y_end = int(ball_center[1] + display_vy * arrow_time)

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

        predicted_x = int(ball_center[0] + display_vx * prediction_time)                
        predicted_y = int(ball_center[1] + display_vy * prediction_time)

        cv2.circle(
            frame,
            (predicted_x, predicted_y),
            6,
            (255, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"Selected Window: {selected_window}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    # 3種類の予測誤差を同時に評価
    if ball_center is not None:
        for window in window_sizes:
            while (
                pending_predictions[window]
                and pending_predictions[window][0][0] <= detection_time
            ):
                target_time, predicted_center, prediction_state = (
                    pending_predictions[window].pop(0)
                )
                error = np.hypot(
                    ball_center[0] - predicted_center[0],
                    ball_center[1] - predicted_center[1]
                )
                prediction_errors[window].append(error)
                all_prediction_errors[window].append(error)
                if prediction_state in state_prediction_errors:
                    state_prediction_errors[prediction_state][window].append(error)

                # STEP 37：運動状態に応じてWindowを選択
                if prediction_state == "Turning":
                    chosen_window = 1
                else:
                    chosen_window = 5

                # 選択したWindowの誤差だけを保存
                if window == chosen_window:
                    adaptive_errors.append(error)
                    adaptive_errors_recent.append(error)
                    adaptive_window_counts[chosen_window] += 1

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
            if ball_circularity is not None:
                print(f"Ball Circularity: {ball_circularity:.3f}")
        else:
            print("Ball not detected")

        print(f"Motion State: {motion_state}")

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

        # 適応方式の1秒平均誤差
        if adaptive_errors_recent:
            mean_adaptive_error = np.mean(adaptive_errors_recent)
            print(
                f"Adaptive Mean Prediction Error: "
                f"{mean_adaptive_error:.2f} px"
            )
            adaptive_errors_recent.clear()
            
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

print("\n=== Motion State Comparison ===")
for state in motion_states:
    print(f"\n{state}:")
    for window in window_sizes:
        errors = state_prediction_errors[state][window]
        if errors:
            mean_error = np.mean(errors)
            print(
                f"Window={window}: "
                f"Predictions={len(errors)}, "
                f"Mean Error={mean_error:.2f} px"
            )
        else:
            print(f"Window={window}: No predictions")

print("\n=== Adaptive Window Comparison ===")
if adaptive_errors:
    adaptive_mean = np.mean(adaptive_errors)
    fixed_mean = np.mean(all_prediction_errors[5])
    print(f"Adaptive Mean Error: {adaptive_mean:.2f} px")
    print(f"Fixed Window=5 Mean Error: {fixed_mean:.2f} px")
    print(f"Adaptive Predictions: {len(adaptive_errors)}")
    print(f"Window=1 selected: {adaptive_window_counts[1]}")
    print(f"Window=5 selected: {adaptive_window_counts[5]}")
    if len(adaptive_errors) != len(all_prediction_errors[5]):
        print("Warning: Prediction counts do not match")

print("\n=== Detection Performance ===")
if total_frames > 0:
    detection_rate = detected_frames / total_frames * 100
    print(f"Total Frames: {total_frames}")
    print(f"Detected Frames: {detected_frames}")
    print(f"Detection Rate: {detection_rate:.2f}%")
else:
    print("No frames captured")

cap.release()
cv2.destroyAllWindows()