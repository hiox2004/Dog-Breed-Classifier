"""
Demo script for dog breed prediction with visual output.
Usage: python predict_demo.py --image path/to/dog.jpg
"""
import argparse
import json
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import patches


def load_model_and_classes(model_path: str, classes_path: str):
    """Load the trained model and class mapping."""
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading classes from {classes_path}...")
    with open(classes_path, 'r') as f:
        classes = json.load(f)
    
    # Convert string keys to integers and create reverse mapping
    class_names = {int(k): v for k, v in classes.items()}
    
    return model, class_names


def preprocess_image(image_path: str, target_size=(299, 299)):
    """Load and preprocess image for prediction."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original for display
    original = img_rgb.copy()
    
    # Resize and normalize for model
    img_resized = cv2.resize(img_rgb, target_size)
    img_normalized = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, original


def predict_breed(model, image_batch, class_names, top_k=3):
    """Make prediction and return top-k results."""
    # Get predictions
    predictions = model.predict(image_batch, verbose=0)
    
    # Get top-k indices
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    
    # Format results
    results = []
    for idx in top_indices:
        breed = class_names[idx]
        confidence = predictions[0][idx] * 100
        results.append({
            'breed': breed,
            'confidence': confidence
        })
    
    return results


def visualize_prediction(original_image, predictions, save_path=None):
    """Create a nice visualization of the prediction."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Display original image
    ax1.imshow(original_image)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=16, fontweight='bold')
    
    # Create prediction bar chart
    breeds = [p['breed'].replace('_', ' ').title() for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    colors = ['#2ecc71', '#3498db', '#95a5a6']  # Green, Blue, Gray
    
    y_pos = np.arange(len(breeds))
    bars = ax2.barh(y_pos, confidences, color=colors, alpha=0.8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(breeds, fontsize=12)
    ax2.invert_yaxis()
    ax2.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Top 3 Predictions', fontsize=16, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    # Add confidence values on bars
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        ax2.text(conf + 2, i, f'{conf:.2f}%', 
                va='center', fontsize=11, fontweight='bold')
    
    # Add grid
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Overall styling
    plt.suptitle('🐕 Dog Breed Prediction', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Prediction visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_predictions(predictions):
    """Print predictions to console in a nice format."""
    print("\n" + "="*60)
    print("🐕 DOG BREED PREDICTION RESULTS".center(60))
    print("="*60)
    
    for i, pred in enumerate(predictions, 1):
        breed = pred['breed'].replace('_', ' ').title()
        confidence = pred['confidence']
        
        # Create confidence bar
        bar_length = int(confidence / 2)  # Scale to 50 chars max
        bar = '█' * bar_length + '░' * (50 - bar_length)
        
        print(f"\n{i}. {breed}")
        print(f"   [{bar}] {confidence:.2f}%")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Predict dog breed from image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='artifacts/best_model.keras',
                       help='Path to trained model')
    parser.add_argument('--classes', type=str, default='artifacts/classes.json',
                       help='Path to classes JSON file')
    parser.add_argument('--output', type=str, default='prediction_result.png',
                       help='Path to save visualization (default: prediction_result.png)')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions to show (default: 3)')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"❌ Error: Image not found at {args.image}")
        return
    
    # Load model and classes
    model, class_names = load_model_and_classes(args.model, args.classes)
    
    # Preprocess image
    print(f"\nProcessing image: {args.image}")
    image_batch, original = preprocess_image(args.image)
    
    # Make prediction
    print("Making prediction...")
    predictions = predict_breed(model, image_batch, class_names, top_k=args.top_k)
    
    # Print to console
    print_predictions(predictions)
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_prediction(original, predictions, save_path=args.output)
    
    print("\n✨ Done!\n")


if __name__ == '__main__':
    main()
