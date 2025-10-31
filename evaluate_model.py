"""
Model Evaluation Script
=======================
Evaluates the trained model on the validation dataset and generates a comprehensive report.
This script provides credible, reproducible evidence of model performance.

Usage:
    python evaluate_model.py --model artifacts/best_model.keras
"""

import os
import json
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import Config
from src.data import load_labels, attach_filepaths, split_train_val, make_generators

# Initialize config
config = Config()


def evaluate_model(model_path, output_dir="evaluation_results"):
    """
    Evaluate model and generate comprehensive report.
    
    Args:
        model_path: Path to the saved model
        output_dir: Directory to save evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print(f"MODEL EVALUATION REPORT")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load model
    print(f"\n[1/6] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"✓ Model loaded successfully")
    print(f"  Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")
    
    # Load data
    print(f"\n[2/6] Loading validation dataset...")
    df = load_labels(config.LABELS_CSV)
    df = attach_filepaths(df, config.TRAIN_DIR)
    train_df, val_df = split_train_val(df, val_size=0.2, seed=config.SEED)
    
    # Get class names (sorted for consistency)
    class_names = sorted(df["breed"].unique().tolist())
    
    # Create validation generator (no augmentation)
    _, val_gen, class_names, _ = make_generators(
        train_df=train_df,
        val_df=val_df,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
        classes=class_names
    )
    
    # Count validation samples
    val_samples = len(val_df)
    print(f"✓ Validation dataset loaded")
    print(f"  Number of samples: {val_samples}")
    print(f"  Number of classes: {len(class_names)}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    
    # Evaluate model
    print(f"\n[3/6] Evaluating model performance...")
    eval_results = model.evaluate(val_gen, verbose=1)
    
    # Handle multiple metrics (loss, accuracy, top3_acc, etc.)
    if isinstance(eval_results, list):
        loss = eval_results[0]
        accuracy = eval_results[1]  # First metric after loss is usually accuracy
    else:
        loss = eval_results
        accuracy = 0.0
    
    print(f"\n✓ Evaluation complete")
    print(f"  Validation Loss: {loss:.4f}")
    print(f"  Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Get predictions for detailed metrics
    print(f"\n[4/6] Generating predictions for detailed analysis...")
    y_true = []
    y_pred = []
    y_pred_probs = []
    
    val_gen.reset()  # Reset generator to start from beginning
    for images, labels in val_gen:
        predictions = model.predict(images, verbose=0)
        y_pred_probs.extend(predictions)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels, axis=1))
        if len(y_true) >= val_samples:  # Stop when we've processed all samples
            break
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_probs = np.array(y_pred_probs)
    
    # Calculate Top-K accuracy
    top3_acc = top_k_accuracy_score(y_true, y_pred_probs, k=3)
    top5_acc = top_k_accuracy_score(y_true, y_pred_probs, k=5)
    
    print(f"✓ Predictions generated")
    print(f"  Top-1 Accuracy: {accuracy*100:.2f}%")
    print(f"  Top-3 Accuracy: {top3_acc*100:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc*100:.2f}%")
    
    # Generate classification report
    print(f"\n[5/6] Generating classification report...")
    class_report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Save detailed metrics to JSON
    results = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "model_architecture": {
            "total_parameters": int(model.count_params()),
            "trainable_parameters": int(trainable_params),
            "non_trainable_parameters": int(non_trainable_params)
        },
        "dataset": {
            "validation_samples": int(val_samples),
            "num_classes": len(class_names),
            "batch_size": config.BATCH_SIZE,
            "image_size": list(config.IMAGE_SIZE)
        },
        "performance_metrics": {
            "validation_loss": float(loss),
            "top1_accuracy": float(accuracy),
            "top3_accuracy": float(top3_acc),
            "top5_accuracy": float(top5_acc),
            "macro_avg_precision": float(class_report['macro avg']['precision']),
            "macro_avg_recall": float(class_report['macro avg']['recall']),
            "macro_avg_f1_score": float(class_report['macro avg']['f1-score']),
            "weighted_avg_precision": float(class_report['weighted avg']['precision']),
            "weighted_avg_recall": float(class_report['weighted avg']['recall']),
            "weighted_avg_f1_score": float(class_report['weighted avg']['f1-score'])
        },
        "per_class_metrics": {
            class_name: {
                "precision": float(metrics['precision']),
                "recall": float(metrics['recall']),
                "f1_score": float(metrics['f1-score']),
                "support": int(metrics['support'])
            }
            for class_name, metrics in class_report.items()
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']
        }
    }
    
    # Save JSON report
    json_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved detailed report: {json_path}")
    
    # Generate and save confusion matrix
    print(f"\n[6/6] Generating confusion matrix visualization...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Create confusion matrix heatmap (simplified for 120 classes)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, cmap='Blues', square=True, cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Validation Accuracy: {accuracy*100:.2f}%\n{timestamp}', 
              fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix: {cm_path}")
    
    # Generate summary text report
    print(f"\n[7/6] Generating text summary...")
    summary_path = os.path.join(output_dir, f"evaluation_summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Path: {model_path}\n\n")
        
        f.write("MODEL ARCHITECTURE:\n")
        f.write(f"  Total Parameters: {model.count_params():,}\n")
        f.write(f"  Trainable Parameters: {trainable_params:,}\n")
        f.write(f"  Non-trainable Parameters: {non_trainable_params:,}\n\n")
        
        f.write("DATASET INFORMATION:\n")
        f.write(f"  Validation Samples: {val_samples:,}\n")
        f.write(f"  Number of Classes: {len(class_names)}\n")
        f.write(f"  Image Size: {config.IMAGE_SIZE}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  Validation Loss: {loss:.4f}\n")
        f.write(f"  Top-1 Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"  Top-3 Accuracy: {top3_acc*100:.2f}%\n")
        f.write(f"  Top-5 Accuracy: {top5_acc*100:.2f}%\n\n")
        
        f.write("AGGREGATE METRICS:\n")
        f.write(f"  Macro Avg Precision: {class_report['macro avg']['precision']:.4f}\n")
        f.write(f"  Macro Avg Recall: {class_report['macro avg']['recall']:.4f}\n")
        f.write(f"  Macro Avg F1-Score: {class_report['macro avg']['f1-score']:.4f}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("This report was generated automatically and can be reproduced\n")
        f.write("by running: python evaluate_model.py --model " + model_path + "\n")
        f.write("=" * 70 + "\n")
    
    print(f"✓ Saved text summary: {summary_path}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   ✓ Validation Accuracy: {accuracy*100:.2f}%")
    print(f"   ✓ Top-3 Accuracy: {top3_acc*100:.2f}%")
    print(f"   ✓ Top-5 Accuracy: {top5_acc*100:.2f}%")
    print(f"\n📁 Generated Files:")
    print(f"   • {json_path}")
    print(f"   • {cm_path}")
    print(f"   • {summary_path}")
    print(f"\n💡 These results can be reproduced at any time by running:")
    print(f"   python evaluate_model.py --model {model_path}")
    print("\n" + "=" * 70 + "\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained model and generate comprehensive report"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="artifacts/best_model.keras",
        help="Path to the trained model file (default: artifacts/best_model.keras)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results (default: evaluation_results)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        exit(1)
    
    evaluate_model(args.model, args.output)
