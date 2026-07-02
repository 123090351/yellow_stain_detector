from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

model = YOLO(ROOT / "models" / "pretrained" / "yolo11n.pt")

results = model.predict(ROOT / "huangban" / "1.png", save=True)

r = results[0]
print("检测到的框数量:", len(r.boxes))
print("boxes:", r.boxes.xyxy)   # 坐标
print("conf:", r.boxes.conf)    # 置信度
print("cls:", r.boxes.cls)      # 类别 id
print("结果图保存在:", r.save_dir)
