import os
from ultralytics import YOLO

def run_freeze_optimization():
    # 22 = Head only
    # 15 = Head + 50% Neck
    # 11 = Head + Full Neck

    freeze_milestones = [22, 15, 11]
    
    results_log = {}
    
    print("Starting Industrial Freeze Optimization Loop...")
    print(f"Dataset path verified via dataset.yaml configuration.")
    print("-" * 60)

    for num_layers in freeze_milestones:
        print(f"\n[LAUNCHING] Experiment: Freezing first {num_layers} layers")
        
        # Always initialize a fresh, pristine pre-trained model for each iteration
        model = YOLO("yolo11n.pt") 
        
        metrics = model.train(
            data="dataset.yaml",
            epochs=30,          # 30 epochs is the standard window to detect overfitting trends
            imgsz=640,          # Native high-resolution processing for defect spotting
            batch=16,
            freeze=num_layers,  # Dynamic architecture slicing
            device=0,           # Forces GPU training (set to 'cpu' if no dedicated GPU)
            workers=4,          # Optimized multi-threaded data loading
            verbose=False,      # Silences step-by-step epoch logs to keep console readable
            project="factory_optimization",
            name=f"freeze_{num_layers}"
        )
        
        # Extract the overall validation fitness metric (composite score of mAP50 and mAP50-95)
        performance_score = metrics.fitness
        results_log[f"freeze_{num_layers}"] = performance_score
        
        print(f"[COMPLETED] freeze={num_layers} | Validation Fitness Score: {performance_score:.4f}")
        print("-" * 60)

    # =====================================================================
    # EXPERIMENT SUMMARY PRINT OUT
    # =====================================================================
    print("\n" + "="*40)
    print("      FINAL EXPERIMENT SUMMARY      ")
    print("="*40)
    print("Configuration    |  Validation Fitness")
    print("-" * 40)
    
    best_config = None
    best_score = -1.0
    
    for run, score in results_log.items():
        print(f"{run:<16} |  {score:.4f}")
        if score > best_score:
            best_score = score
            best_config = run
            
    print("-" * 40)
    print(f"RECOMMENDED DEPLOYMENT SELECTION: {best_config.upper()} ({best_score:.4f})")
    print("="*40)

if __name__ == "__main__":
    run_freeze_optimization()
