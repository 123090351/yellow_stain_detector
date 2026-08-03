import os
import csv
from pathlib import Path
from ultralytics import YOLO

def classify_unlabeled_images(
    model_path, 
    data_path, 
    conf_thresh=0.175, 
    output_csv="classification_results.csv",
    save_annotated=True,          # <--- Flag to enable saving visual outputs
    annotated_dir="/mnt/output/annotated" # <--- Root directory for saved outputs
):
    """
    Runs YOLO inference on an unlabeled image dataset, classifies each image as 'Defective' or 'OK',
    and saves the visual images with bounding boxes into separate 'OK' and 'Defective' subfolders.
    """
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Determine image directory path
    images_dir = data_path

    # Define and create subfolders for OK and Defective predictions
    ok_dir = os.path.join(annotated_dir, "OK")
    defective_dir = os.path.join(annotated_dir, "Defective")

    if save_annotated:
        os.makedirs(ok_dir, exist_ok=True)
        os.makedirs(defective_dir, exist_ok=True)

    print("=" * 60)
    print(f"Targeting Images Directory : {images_dir}")
    if save_annotated:
        print(f"Saving Classified Images To : {annotated_dir} (OK / Defective)")

    # Manual check of image count
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    actual_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]
    print(f"Actual image files on disk : {len(actual_files)}")
    print("=" * 60)

    defective_count = 0
    ok_count = 0
    results_list = []

    # Stream predictions over images
    results = model.predict(source=images_dir, conf=conf_thresh, stream=True, verbose=False)

    for result in results:
        img_name = os.path.basename(result.path)
        
        # Check if model detected any bounding boxes
        num_detections = len(result.boxes)
        status = "Defective" if num_detections > 0 else "OK"

        if status == "Defective":
            defective_count += 1
            max_conf = float(result.boxes.conf.max()) if num_detections > 0 else 0.0
            print(f"[DEFECT DETECTED] {img_name} -> {num_detections} defect(s) found (Max Conf: {max_conf:.2f})")
            target_subfolder = defective_dir
        else:
            ok_count += 1
            max_conf = 0.0
            target_subfolder = ok_dir

        # --- SAVE ANNOTATED IMAGE TO SPECIFIC SUBFOLDER ---
        if save_annotated:
            save_path = os.path.join(target_subfolder, img_name)
            result.save(filename=save_path)  # Draws boxes (if any) and saves to OK or Defective folder

        results_list.append({
            "image_name": img_name,
            "status": status,
            "num_defects": num_detections,
            "max_confidence": round(max_conf, 4)
        })

    # Save summary report to CSV
    if output_csv:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_name", "status", "num_defects", "max_confidence"])
            writer.writeheader()
            writer.writerows(results_list)
        print(f"\nResults successfully saved to: {output_csv}")

    total_images = defective_count + ok_count
    defect_rate = (defective_count / total_images * 100) if total_images > 0 else 0.0

    return {
        "total_images": total_images,
        "defective_count": defective_count,
        "ok_count": ok_count,
        "defect_rate_pct": defect_rate
    }


if __name__ == "__main__":
    SAVED_MODEL_PATH = "/mnt/huangban-script/runs/detect/factory_optimization/freeze_3-42/weights/best.pt"
    DATA_PATH = "/mnt/huangban-test"  # Direct folder containing target images

    # Execute classification
    summary = classify_unlabeled_images(
        model_path=SAVED_MODEL_PATH,
        data_path=DATA_PATH,
        conf_thresh=0.175,
        output_csv="test_results.csv",
        save_annotated=True,
        annotated_dir="/mnt/output/annotated"
    )

    print("\n" + "=" * 60)
    print("               UNLABELED CLASSIFICATION SUMMARY            ")
    print("=" * 60)
    print(f"Total Processed Images : {summary['total_images']}")
    print(f"Defective Images Found : {summary['defective_count']}")
    print(f"OK (Clean) Images Found: {summary['ok_count']}")
    print(f"Defect Rate            : {summary['defect_rate_pct']:.2f}%")
    print("=" * 60)