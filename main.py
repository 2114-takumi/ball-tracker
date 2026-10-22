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

            x, y = center

            x = int(x)
            y = int(y)
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

    # ⑪ FPS測定
    current_time = time.perf_counter()
    elapsed_time = current_time - start_time

    if elapsed_time >= 1.0:

        fps = frame_count / elapsed_time

        print(f"FPS: {fps:.2f}")
        print(f"Largest Contour Area: {area:.1f}")

        if ball_center is not None:
            print(f"Ball Center: {ball_center}")
            print(f"Ball Radius: {ball_radius}")
        else:
            print("Ball not detected")

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
cap.release()
cv2.destroyAllWindows()