import os
import torch
from ultralytics import YOLO

def evaluate_image_level(model, data_dir, conf_thresh=0.175):
    """
    Evaluates image-level (pass/fail) precision, recall, accuracy, and fitness.
    - Ground Truth Positive: The image's corresponding .txt file has >= 1 defect line.
    - Model Predicted Positive: YOLO outputs >= 1 bounding box above `conf_thresh`.
    """
    val_images_dir = os.path.join(data_dir, "huangban_v1/images/val")
    val_labels_dir = os.path.join(data_dir, "huangban_v1/labels/val")

    # If the subfolder pathing differs, fall back to scanning labels/val directory directly
    if not os.path.exists(val_images_dir):
        val_images_dir = os.path.join(data_dir, "images/val")
        val_labels_dir = os.path.join(data_dir, "labels/val")

    tp, fp, tn, fn = 0, 0, 0, 0

    # Stream prediction over validation images for minimal GPU memory overhead
    results = model.predict(source=val_images_dir, conf=conf_thresh, stream=True, verbose=False)

    for result in results:
        img_name = os.path.basename(result.path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(val_labels_dir, label_name)

        # Check Ground Truth: Does the label file exist and contain at least 1 object line?
        has_gt_defect = False
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                if len(lines) > 0:
                    has_gt_defect = True

        # Check Model Prediction: Did YOLO find at least 1 defect box?
        has_pred_defect = len(result.boxes) > 0

        # Confusion Matrix Logic
        if has_gt_defect and has_pred_defect:
            tp += 1
        elif not has_gt_defect and has_pred_defect:
            fp += 1
        elif not has_gt_defect and not has_pred_defect:
            tn += 1
        elif has_gt_defect and not has_pred_defect:
            fn += 1

    # Calculate Image-Level Metrics
    img_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    img_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    img_accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    
    # Image-Level Fitness (60% Recall for factory defect safety + 40% Precision)
    img_fitness = (0.6 * img_recall) + (0.4 * img_precision)

    return {
        "img_precision": img_precision,
        "img_recall": img_recall,
        "img_accuracy": img_accuracy,
        "img_fitness": img_fitness,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


def run_freeze_optimization():
    freeze_milestones = [3]
    data_path = "/mnt/huangban-data/"
    results_log = {}
    
    print("Starting Industrial Freeze Optimization Loop with Image-Level Tracking...")

    for num_layers in freeze_milestones:
        print(f"\nFreezing first {num_layers} layers")
        
        # Always initialize a fresh, pristine pre-trained model
        model = YOLO("yolo11n.pt") 
        
        metrics = model.train(
            data=data_path,
            epochs=200,
            imgsz=640,
            batch=16,
            freeze=num_layers,
            patience=15,
            device=0,
            workers=12,
            seed=42,
            dropout=0.0,
            
            # Disabled spatial/color augmentations
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0,
            mosaic=0.0, close_mosaic=0, erasing=0.0,
            verbose=False,
            project="factory_optimization",
            name=f"freeze_{num_layers}"
        )
        
        box_fitness = float(metrics.fitness)
        box_precision = float(metrics.box.p.squeeze().item())
        box_recall = float(metrics.box.r.squeeze().item())
        
        # Image-Level Metrics (Pass/Fail Evaluation)
        best_model_path = os.path.join(metrics.save_dir, "weights", "best.pt")
        trained_model = YOLO(best_model_path)
        
        img_metrics = evaluate_image_level(trained_model, data_path, conf_thresh=0.175)
        
        results_log[f"freeze_{num_layers}"] = {
            "box_fitness": box_fitness,
            "box_p": box_precision,
            "box_r": box_recall,
            "img_p": float(img_metrics["img_precision"]),
            "img_r": float(img_metrics["img_recall"]),
            "img_acc": float(img_metrics["img_accuracy"]),
            "img_fitness": float(img_metrics["img_fitness"])
        }

    # EXPERIMENT SUMMARY PRINT OUT
    print("\n" + "="*85)
    print("                              FINAL EXPERIMENT SUMMARY                              ")
    print("="*85)
    print(f"{'Config':<12} | {'Box P':<7} | {'Box R':<7} | {'Box Fit':<8} | {'Img P':<7} | {'Img R':<7} | {'Img Acc':<8} | {'Img Fit':<8}")
    print("-" * 85)
    
    best_config = None
    best_score = -1.0
    
    for run, m in results_log.items():
        print(f"{run:<12} | {m['box_p']:<7.4f} | {m['box_r']:<7.4f} | {m['box_fitness']:<8.4f} | {m['img_p']:<7.4f} | {m['img_r']:<7.4f} | {m['img_acc']:<8.4f} | {m['img_fitness']:<8.4f}")
        
        if m["img_fitness"] > best_score:
            best_score = m["img_fitness"]
            best_config = run


if __name__ == "__main__":
    run_freeze_optimization()