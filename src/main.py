import cv2
from ultralytics import YOLO

# 学習済みのモデルをロード
model = YOLO('yolov8n.pt')

# 動画ファイルのパスを指定
video_path = "../data/soccer.mp4"
cap = cv2.VideoCapture(video_path)

# 処理した動画を保存するための設定をします
output_path = "../output/soccer.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 動画の形式をMP4に指定
fps = cap.get(cv2.CAP_PROP_FPS)          # 元動画のFPS（速さ）を取得
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 動画の幅を取得
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 動画の高さを取得
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 動画が終わるまで繰り返し処理します
while cap.isOpened():
    # 動画から1枚の画像（フレーム）を読み込みます
    success, frame = cap.read()

    if success:  # 画像の読み込みに成功したら
        # YOLOv8で物体を追跡します
        # persist=Trueで物体の動きを追跡し続けます
        results = model.track(frame, persist=True)

        # 検出結果を画像に描画します
        annotated_frame = results[0].plot()

        # 処理した画像を出力ファイルに保存します
        out.write(annotated_frame)

        # 処理中の画像をウィンドウで表示します
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        
        # キーボードが押されたかチェック（1ミリ秒待機）
        key = cv2.waitKey(1)
        if key != -1:  # キーが押されたら
            print("STOP PLAY")
            break  # 処理を終了

# リソースの解放
cap.release()         # ビデオの読み込みを終了
out.release()         # ビデオの書き込みを終了
cv2.destroyAllWindows()  # 表示していたウィンドウを閉じる