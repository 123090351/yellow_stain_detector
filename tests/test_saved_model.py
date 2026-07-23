import os
from ultralytics import YOLO

def evaluate_saved_model(model_path, data_path, conf_thresh=0.175, split="test"):
    """
    Loads a pre-trained/saved YOLO model and evaluates it directly on test/val data.
    """
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    images_dir = os.path.join(data_path, f"huangban_v2/images/{split}")
    labels_dir = os.path.join(data_path, f"huangban_v2/labels/{split}")

    if not os.path.exists(images_dir):
        images_dir = os.path.join(data_path, f"images/{split}")
        labels_dir = os.path.join(data_path, f"labels/{split}")

    tp, fp, tn, fn = 0, 0, 0, 0

    # Insert this check inside evaluate_saved_model before running predict:
    print("="*50)
    print(f"Targeting Images Directory : {images_dir}")
    print(f"Targeting Labels Directory : {labels_dir}")

# Count exact image files in folder manually
    actual_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"Actual image files on disk : {len(actual_files)}")
    print("="*50)

    # Stream prediction over test images
    results = model.predict(source=images_dir, conf=conf_thresh, stream=True, verbose=False)

    for result in results:
        img_name = os.path.basename(result.path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        if not os.path.exists(label_path):
            continue

        has_gt_defect = False
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            if len(lines) > 0:
                has_gt_defect = True

        has_pred_defect = len(result.boxes) > 0

        if has_gt_defect and has_pred_defect:
            tp += 1
        elif not has_gt_defect and has_pred_defect:
            fp += 1
        elif not has_gt_defect and not has_pred_defect:
            tn += 1
        elif has_gt_defect and not has_pred_defect:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    fitness = (0.6 * recall) + (0.4 * precision)

    return {
        "img_precision": precision,
        "img_recall": recall,
        "img_accuracy": accuracy,
        "img_fitness": fitness,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }

if __name__ == "__main__":
    # Path to your saved model checkpoint from a previous run
    SAVED_MODEL_PATH = "/mnt/huangban-script/runs/detect/factory_optimization/freeze_3-23/weights/best.pt"
    DATA_PATH = "/mnt/huangban-data/"

    # Evaluate on TEST set directly without retraining
    metrics = evaluate_saved_model(
        model_path=SAVED_MODEL_PATH, 
        data_path=DATA_PATH, 
        conf_thresh=0.175, 
        split="test"
    )

    print("\n" + "="*50)
    print("           DIRECT TEST EVALUATION METRICS          ")
    print("="*50)
    print(f"Precision : {metrics['img_precision']:.4f}")
    print(f"Recall    : {metrics['img_recall']:.4f}")
    print(f"Accuracy  : {metrics['img_accuracy']:.4f}")
    print(f"Fitness   : {metrics['img_fitness']:.4f}")
    print(f"TP: {metrics['tp']} | FP: {metrics['fp']} | TN: {metrics['tn']} | FN: {metrics['fn']}")
