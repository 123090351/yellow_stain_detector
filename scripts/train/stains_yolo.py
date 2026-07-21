import os
import torch
from ultralytics import YOLO

def evaluate_image_level(model, data_yaml, conf_thresh=0.25):
    """
    Evaluates image-level (pass/fail) precision, recall, accuracy, and fitness.
    An image is considered a ground-truth Positive if it has >= 1 labeled defect box.
    An image is predicted Positive if YOLO outputs >= 1 detection box above `conf_thresh`.
    """
    # Run validation mode to get predictions per image
    val_results = model.val(data=data_yaml, split='val', verbose=False, conf=conf_thresh)
    
    tp, fp, tn, fn = 0, 0, 0, 0
    
    # Process each image in the validation dataset
    for img_info in val_results.nt_per_image:  # or iterate validation batch outputs
        pass  # Handled below via val dataloader inspection

    # Evaluate predictions directly from validation loader outputs
    val_loader = val_results.dataloader
    
    for batch in val_loader:
        img_paths = batch['im_file']
        targets = batch['cls']          # Ground truth classes
        batch_idx = batch['batch_idx']  # Image indices within batch
        
        # Run inference on current image batch
        preds = model(batch['img'], verbose=False, conf=conf_thresh)
        
        for i, img_path in enumerate(img_paths):
            # Check ground truth: Does this image have at least 1 defect box?
            has_gt_defect = (batch_idx == i).sum().item() > 0
            
            # Check model prediction: Did YOLO predict at least 1 box?
            has_pred_defect = len(preds[i].boxes) > 0
            
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
    
    # Image-Level Fitness (weighted composite: 60% Recall for factory safety + 40% Precision)
    img_fitness = (0.6 * img_recall) + (0.4 * img_precision)
    
    return {
        "img_precision": img_precision,
        "img_recall": img_recall,
        "img_accuracy": img_accuracy,
        "img_fitness": img_fitness,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


def run_freeze_optimization():
    freeze_milestones = [0]
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
            patience=30,
            device=0,
            workers=12,
            seed=42,
            dropout=0.0,
            cls=1.5,
            
            # Disabled spatial/color augmentations
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0,
            mosaic=0.0, close_mosaic=0, erasing=0.0,
            verbose=False,
            project="factory_optimization",
            name=f"freeze_{num_layers}"
        )
        
        box_fitness = metrics.fitness
        box_precision = metrics.box.p
        box_recall = metrics.box.r
        
        # Image-Level Metrics (Pass/Fail Evaluation)
        print("Evaluating Image-Level (Yield) Performance...")
        best_model_path = os.path.join(metrics.save_dir, "weights", "best.pt")
        trained_model = YOLO(best_model_path)
        
        img_metrics = evaluate_image_level(trained_model, data_path, conf_thresh=0.25)
        
        results_log[f"freeze_{num_layers}"] = {
            "box_fitness": box_fitness,
            "box_p": box_precision,
            "box_r": box_recall,
            "img_p": img_metrics["img_precision"],
            "img_r": img_metrics["img_recall"],
            "img_acc": img_metrics["img_accuracy"],
            "img_fitness": img_metrics["img_fitness"]
        }

        print(f"freeze={num_layers} | Box Fitness: {box_fitness:.4f} | Image Fitness: {img_metrics['img_fitness']:.4f}")

    # EXPERIMENT SUMMARY PRINT OUT
    print("\n" + "="*85)
    print("                              FINAL EXPERIMENT SUMMARY                              ")
    print("="*85)
    print(f"{'Config':<12} | {'Box P':<7} | {'Box R':<7} | {'Box Fit':<8} | {'Img P':<7} | {'Img R':<7} | {'Img Acc':<8} | {'Img Fit':<8}")
    print("-" * 85)
    
    best_config = None
    best_score = -1.0
    
    for run, m in results_log.items():
        print(f"{run:<12} | {m['box_fitness']:<8.4f} | {m['box_p']:<7.4f} | {m['box_r']:<7.4f} | {m['img_acc']:<8.4f} | {m['img_p']:<7.4f} | {m['img_r']:<7.4f} | {m['img_fitness']:<8.4f}")
        
        if m["img_fitness"] > best_score:
            best_score = m["img_fitness"]
            best_config = run
            
    print("-" * 85)
    print(f"RECOMMENDED DEPLOYMENT SELECTION (BY IMAGE FITNESS): {best_config.upper()} ({best_score:.4f})")


if __name__ == "__main__":
    run_freeze_optimization()