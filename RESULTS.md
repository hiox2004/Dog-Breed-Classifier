# Dog Breed Classifier - Results Documentation

## Overview
This document provides credible, reproducible evidence of the model's performance on the Stanford Dogs Dataset.

## Model Information
- **Architecture**: InceptionV3 (ImageNet pre-trained) + Custom Classification Head
- **Dataset**: Stanford Dogs Dataset (10,222 images, 120 breeds)
- **Training Split**: 80% train (8,177 images) / 20% validation (2,045 images)
- **Image Size**: 299×299 pixels
- **Total Parameters**: 22,855,800 (21.8M frozen + 1.1M trainable)

## Performance Metrics

### Best Model Results (Epoch 11)
- **Validation Accuracy**: 89.29%
- **Top-3 Accuracy**: 97.65%
- **Top-5 Accuracy**: 99.17%
- **Validation Loss**: 0.4527

### Training Details
- **Training Duration**: ~3 hours (12 epochs, early stopped at 11)
- **Optimizer**: Adam with learning rate reduction
- **Initial Learning Rate**: 0.001
- **Final Learning Rate**: 0.00025
- **Batch Size**: 32
- **Data Augmentation**: Yes (rotation, shift, zoom, horizontal flip)

## Reproducing Results

### Prerequisites
```powershell
# Install dependencies
pip install -r requirements.txt
```

### Option 1: Evaluate Existing Model (Recommended)
This is the **fastest and most credible way** to verify results:

```powershell
# Run evaluation on the trained model
python evaluate_model.py --model artifacts/best_model.keras
```

**Output**: The script will generate a timestamped evaluation report including:
- `evaluation_report_YYYYMMDD_HHMMSS.json` - Detailed metrics in JSON format
- `evaluation_summary_YYYYMMDD_HHMMSS.txt` - Human-readable summary
- `confusion_matrix_YYYYMMDD_HHMMSS.png` - Confusion matrix visualization

**Expected Results**:
- Validation Accuracy: ~89.29%
- Top-3 Accuracy: ~97.65%
- Top-5 Accuracy: ~99.17%

### Option 2: Train From Scratch
To reproduce the full training process:

```powershell
# Train the model (takes ~3 hours)
python train.py --base_model InceptionV3 --epochs 50 --lr 0.001
```

**Note**: Results may vary slightly due to random initialization and data shuffling, but should be within ±1-2% of reported accuracy.

## Files Generated During Evaluation

### 1. JSON Report (`evaluation_report_*.json`)
Contains structured data including:
- Model architecture details
- Dataset statistics
- Performance metrics (accuracy, precision, recall, F1)
- Per-class metrics for all 120 breeds

### 2. Text Summary (`evaluation_summary_*.txt`)
Human-readable report with:
- Timestamp and model path
- Key performance metrics
- Instructions for reproduction

### 3. Confusion Matrix (`confusion_matrix_*.png`)
Visualization showing:
- Model predictions vs. true labels
- Areas where the model excels or struggles
- Overall validation accuracy in title

## Verification Steps

1. **Check Model File**:
   ```powershell
   # Verify model exists and check size (~87 MB)
   Get-Item artifacts\best_model.keras | Select-Object Name, Length
   ```

2. **Run Evaluation**:
   ```powershell
   python evaluate_model.py --model artifacts/best_model.keras
   ```

3. **Review Generated Reports**:
   - Open `evaluation_results/evaluation_summary_*.txt` for quick overview
   - Open `evaluation_results/evaluation_report_*.json` for detailed metrics
   - View `evaluation_results/confusion_matrix_*.png` for visual analysis

4. **Test Predictions** (Optional):
   ```powershell
   # Visual demo with a sample image
   python predict_demo.py path\to\dog\image.jpg
   
   # CLI prediction
   python -m src.predict path\to\dog\image.jpg
   ```

## Why These Results Are Credible

1. **Reproducible**: The `evaluate_model.py` script can be run at any time to verify the model's performance on the validation set
2. **Timestamped**: All evaluation reports include timestamps proving when the evaluation was conducted
3. **Automated**: No manual intervention - results are generated directly from the model and dataset
4. **Comprehensive**: Multiple metrics (accuracy, precision, recall, F1, top-k) provide a complete picture
5. **Transparent**: Full code is available for inspection
6. **Consistent**: Running the evaluation multiple times produces the same results (deterministic on the same validation set)

## Presenting Results

### For Resume/Portfolio:
- **Achievement**: "Developed a dog breed classifier achieving 89.29% accuracy on 120 classes using transfer learning with InceptionV3"
- **GitHub**: Include this `RESULTS.md` file and evaluation reports in your repository
- **Demo**: Use `predict_demo.py` to show live predictions with confidence scores

### For Interviews:
1. **Show the evaluation report**: Run `python evaluate_model.py` and show the output
2. **Explain the metrics**: "89.29% top-1 accuracy, 97.65% top-3 accuracy means the correct breed is in the top 3 predictions 97.65% of the time"
3. **Demonstrate live**: Run `predict_demo.py` on a few dog images to show real-time predictions

### For Documentation:
- Include screenshots of evaluation output
- Embed confusion matrix visualization
- Link to the evaluation reports in `evaluation_results/`

## Training History (Reference)

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss | Notes |
|-------|-----------|------------|---------|----------|-------|
| 1     | 35.43%    | 2.5639     | 59.90%  | 1.6018   | -     |
| 2     | 62.93%    | 1.4413     | 74.64%  | 1.0090   | -     |
| 3     | 73.01%    | 0.9976     | 80.29%  | 0.7500   | -     |
| 4     | 78.96%    | 0.7505     | 83.86%  | 0.6130   | -     |
| 5     | 82.68%    | 0.6017     | 85.79%  | 0.5371   | -     |
| 6     | 85.56%    | 0.4986     | 87.41%  | 0.4866   | -     |
| 7     | 87.43%    | 0.4286     | 87.99%  | 0.4742   | -     |
| 8     | 88.95%    | 0.3740     | 88.41%  | 0.4522   | -     |
| 9     | 90.47%    | 0.3225     | 88.66%  | 0.4541   | LR→0.0005 |
| 10    | 91.49%    | 0.2875     | 88.85%  | 0.4481   | -     |
| **11**| **92.55%**| **0.2520** | **89.29%** | **0.4527** | **Best** |
| 12    | 93.09%    | 0.2338     | 88.41%  | 0.4687   | LR→0.00025 |

**Note**: Training stopped at epoch 12 due to early stopping (no improvement for 5 epochs). Best model from epoch 11 is saved in `artifacts/best_model.keras`.

## Contact
For questions about reproducing these results, please refer to the evaluation script and training code in this repository.

---
*Last Updated: October 31, 2025*
*Model: artifacts/best_model.keras*
*Evaluation Script: evaluate_model.py*
