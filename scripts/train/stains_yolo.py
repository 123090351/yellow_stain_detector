import os
from ultralytics import YOLO

def run_freeze_optimization():
    # 23 = Head
    # 10-22 = Neck
    # 0-9 = Backbone

    freeze_milestones = [23, 22, 19, 16, 13, 9]
    
    results_log = {}
    
    print("Starting Industrial Freeze Optimization Loop...")

    for num_layers in freeze_milestones:
        print(f"\nFreezing first {num_layers} layers")
        
        # Always initialize a fresh, pristine pre-trained model for each iteration
        model = YOLO("yolo11n.pt") 
        
        metrics = model.train(
            data="/mnt/huangban-data/",
            epochs=30,          # 30 epochs is the standard window to detect overfitting trends
            imgsz=640,          # Native high-resolution processing for defect spotting
            batch=16,
            freeze=num_layers,  # Dynamic architecture slicing
            patience=5,         # Early stopping patience to prevent overfitting
            # device='cpu',       # CPU training
            device=0,           # Forces GPU training
            # workers=4,          # Optimized multi-threaded data loading
            verbose=False,      # Silences step-by-step epoch logs to keep console readable
            project="factory_optimization",
            name=f"freeze_{num_layers}"
        )
        
        # Extract the overall validation fitness metric (composite score of mAP50 and mAP50-95)
        performance_score = metrics.fitness
        results_log[f"freeze_{num_layers}"] = performance_score
        '''
        import random
        performance_score = random.uniform(0.5, 0.95)
        results_log[f"freeze_{num_layers}"] = performance_score
        '''
        print(f"freeze={num_layers} | Validation Fitness Score: {performance_score:.4f}")

    # EXPERIMENT SUMMARY PRINT OUT
    print("\n      FINAL EXPERIMENT SUMMARY      ")
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

if __name__ == "__main__":
    run_freeze_optimization()
