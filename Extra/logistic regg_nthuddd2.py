import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve)
import cv2
import os
from glob import glob
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 60)
print("NTHU-DDD DROWSINESS DETECTION - LOGISTIC REGRESSION")
print("=" * 60)

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================

# Define paths - adjust these based on your Kaggle dataset structure
BASE_PATH = '/kaggle/input/nthuddd2/train_data'  # Update this path
DROWSY_PATH = os.path.join(BASE_PATH, 'drowsy')  # or 'Drowsy', 'sleepy', etc.
ALERT_PATH = os.path.join(BASE_PATH, 'notdrowsy')    # or 'Alert', 'awake', etc.

# Image parameters
IMG_SIZE = (64, 64)  # Reduced size for faster processing with logistic regression
TARGET_SIZE = IMG_SIZE[0] * IMG_SIZE[1] * 3  # Flattened feature size

def load_images_from_folder(folder_path, label):
    """Load images from folder and assign labels"""
    images = []
    labels = []
    
    # Support multiple image formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob(os.path.join(folder_path, '**', ext), recursive=True))
    
    print(f"Found {len(image_paths)} images in {folder_path}")
    
    for img_path in tqdm(image_paths, desc=f"Loading {folder_path.split('/')[-1]} images"):
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                images.append(img)
                labels.append(label)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    return np.array(images), np.array(labels)

# Load dataset
print("\n[1] Loading Dataset...")
print("-" * 60)

# Load drowsy images (label = 1)
drowsy_images, drowsy_labels = load_images_from_folder(DROWSY_PATH, 1)

# Load alert images (label = 0)
alert_images, alert_labels = load_images_from_folder(ALERT_PATH, 0)

# Combine data
X = np.concatenate([alert_images, drowsy_images], axis=0)
y = np.concatenate([alert_labels, drowsy_labels], axis=0)

print(f"\nTotal images loaded: {len(X)}")
print(f"Alert images: {np.sum(y == 0)}")
print(f"Drowsy images: {np.sum(y == 1)}")
print(f"Image shape: {X[0].shape}")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n[2] Exploratory Data Analysis...")
print("-" * 60)

# Class distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Class distribution pie chart
class_counts = [np.sum(y == 0), np.sum(y == 1)]
axes[0].pie(class_counts, labels=['Alert', 'Drowsy'], autopct='%1.1f%%', 
            colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')

# Class distribution bar chart
axes[1].bar(['Alert', 'Drowsy'], class_counts, color=['#2ecc71', '#e74c3c'])
axes[1].set_ylabel('Number of Images', fontsize=12)
axes[1].set_title('Sample Count by Class', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Sample images
sample_alert = X[y == 0][0]
sample_drowsy = X[y == 1][0]
combined = np.hstack([sample_alert, sample_drowsy])
axes[2].imshow(combined)
axes[2].set_title('Sample Images: Alert (Left) | Drowsy (Right)', 
                  fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=300, bbox_inches='tight')
plt.show()

# Display sample images from each class
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Sample Images from Dataset', fontsize=16, fontweight='bold')

for i in range(5):
    axes[0, i].imshow(X[y == 0][i])
    axes[0, i].set_title('Alert', color='green')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(X[y == 1][i])
    axes[1, i].set_title('Drowsy', color='red')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
plt.show()

# Pixel intensity distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

alert_mean = X[y == 0].mean(axis=0)
drowsy_mean = X[y == 1].mean(axis=0)

axes[0].imshow(alert_mean.astype(np.uint8))
axes[0].set_title('Average Alert Face', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(drowsy_mean.astype(np.uint8))
axes[1].set_title('Average Drowsy Face', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('average_faces.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================

print("\n[3] Data Preprocessing...")
print("-" * 60)

# Normalize pixel values
X_normalized = X.astype('float32') / 255.0

# Flatten images for logistic regression
X_flattened = X_normalized.reshape(len(X_normalized), -1)
print(f"Flattened shape: {X_flattened.shape}")

# Split into train, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_flattened, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
)  # 0.176 * 0.85 ≈ 0.15, giving us 70-15-15 split

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nData preprocessing completed!")
print(f"Feature dimension: {X_train_scaled.shape[1]}")

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

print("\n[4] Training Logistic Regression Model...")
print("-" * 60)

# Train logistic regression model
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
    C=1.0,
    class_weight='balanced',  # Handle class imbalance
    verbose=1
)

model.fit(X_train_scaled, y_train)

print("\nModel training completed!")
print(f"Number of iterations: {model.n_iter_[0]}")
print(f"Intercept: {model.intercept_[0]:.4f}")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

print("\n[5] Model Evaluation...")
print("-" * 60)

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics for all sets
def calculate_metrics(y_true, y_pred, y_proba, set_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    
    print(f"\n{set_name} Set Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    
    return acc, prec, rec, f1, auc

train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba, "Training")
val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba, "Validation")
test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba, "Test")

# Classification report
print("\n" + "=" * 60)
print("DETAILED CLASSIFICATION REPORT (Test Set)")
print("=" * 60)
print(classification_report(y_test, y_test_pred, 
                          target_names=['Alert', 'Drowsy']))

# ============================================================================
# 6. VISUALIZATION OF RESULTS
# ============================================================================

print("\n[6] Generating Visualizations...")
print("-" * 60)

# Metrics comparison across sets
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
train_vals = train_metrics
val_vals = val_metrics
test_vals = test_metrics

# Bar plots for each metric
for idx, metric in enumerate(metrics_names):
    row = idx // 3
    col = idx % 3
    
    values = [train_vals[idx], val_vals[idx], test_vals[idx]]
    bars = axes[row, col].bar(['Train', 'Val', 'Test'], values, 
                               color=['#3498db', '#f39c12', '#2ecc71'])
    axes[row, col].set_ylabel('Score', fontsize=11)
    axes[row, col].set_title(metric, fontsize=13, fontweight='bold')
    axes[row, col].set_ylim([0, 1.05])
    axes[row, col].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=10)

# Overall comparison
axes[1, 2].plot(['Train', 'Val', 'Test'], 
                [train_vals[0], val_vals[0], test_vals[0]], 
                marker='o', linewidth=2, markersize=8, label='Accuracy')
axes[1, 2].plot(['Train', 'Val', 'Test'], 
                [train_vals[3], val_vals[3], test_vals[3]], 
                marker='s', linewidth=2, markersize=8, label='F1-Score')
axes[1, 2].plot(['Train', 'Val', 'Test'], 
                [train_vals[4], val_vals[4], test_vals[4]], 
                marker='^', linewidth=2, markersize=8, label='AUC-ROC')
axes[1, 2].set_ylabel('Score', fontsize=11)
axes[1, 2].set_title('Performance Comparison', fontsize=13, fontweight='bold')
axes[1, 2].set_ylim([0, 1.05])
axes[1, 2].legend(fontsize=10)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (y_true, y_pred, title) in enumerate([
    (y_train, y_train_pred, 'Training Set'),
    (y_val, y_val_pred, 'Validation Set'),
    (y_test, y_test_pred, 'Test Set')
]):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Alert', 'Drowsy'],
                yticklabels=['Alert', 'Drowsy'],
                ax=axes[idx], cbar_kws={'label': 'Count'})
    axes[idx].set_ylabel('True Label', fontsize=12)
    axes[idx].set_xlabel('Predicted Label', fontsize=12)
    axes[idx].set_title(f'Confusion Matrix - {title}', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC Curve
fig, ax = plt.subplots(figsize=(10, 8))

for y_true, y_proba, label, color in [
    (y_train, y_train_proba, f'Train (AUC = {train_metrics[4]:.3f})', '#3498db'),
    (y_val, y_val_proba, f'Val (AUC = {val_metrics[4]:.3f})', '#f39c12'),
    (y_test, y_test_proba, f'Test (AUC = {test_metrics[4]:.3f})', '#2ecc71')
]:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax.plot(fpr, tpr, linewidth=2, label=label, color=color)

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Precision-Recall Curve
fig, ax = plt.subplots(figsize=(10, 8))

for y_true, y_proba, label, color in [
    (y_train, y_train_proba, 'Train', '#3498db'),
    (y_val, y_val_proba, 'Val', '#f39c12'),
    (y_test, y_test_proba, 'Test', '#2ecc71')
]:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax.plot(recall, precision, linewidth=2, label=label, color=color)

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Prediction distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram of prediction probabilities
axes[0].hist(y_test_proba[y_test == 0], bins=30, alpha=0.6, 
             label='Alert', color='green', edgecolor='black')
axes[0].hist(y_test_proba[y_test == 1], bins=30, alpha=0.6, 
             label='Drowsy', color='red', edgecolor='black')
axes[0].set_xlabel('Predicted Probability (Drowsy)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Predicted Probabilities', 
                  fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

# Box plot
data_to_plot = [y_test_proba[y_test == 0], y_test_proba[y_test == 1]]
bp = axes[1].boxplot(data_to_plot, labels=['Alert', 'Drowsy'], 
                     patch_artist=True, showmeans=True)
bp['boxes'][0].set_facecolor('green')
bp['boxes'][1].set_facecolor('red')
axes[1].set_ylabel('Predicted Probability (Drowsy)', fontsize=12)
axes[1].set_title('Prediction Probability Distribution by Class', 
                  fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 7. SAVE MODEL AND ARTIFACTS
# ============================================================================

print("\n[7] Saving Model and Artifacts...")
print("-" * 60)

# Save model
with open('drowsiness_logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved: drowsiness_logistic_model.pkl")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved: scaler.pkl")

# Save metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Train': train_metrics,
    'Validation': val_metrics,
    'Test': test_metrics
})
metrics_df.to_csv('model_metrics.csv', index=False)
print("✓ Metrics saved: model_metrics.csv")

# Save model summary
with open('model_summary.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("DROWSINESS DETECTION MODEL SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model: Logistic Regression\n")
    f.write(f"Image Size: {IMG_SIZE}\n")
    f.write(f"Total Samples: {len(X)}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Validation Samples: {len(X_val)}\n")
    f.write(f"Test Samples: {len(X_test)}\n\n")
    f.write("Test Set Performance:\n")
    f.write(f"  Accuracy:  {test_metrics[0]:.4f}\n")
    f.write(f"  Precision: {test_metrics[1]:.4f}\n")
    f.write(f"  Recall:    {test_metrics[2]:.4f}\n")
    f.write(f"  F1-Score:  {test_metrics[3]:.4f}\n")
    f.write(f"  AUC-ROC:   {test_metrics[4]:.4f}\n")

print("✓ Summary saved: model_summary.txt")

print("\n" + "=" * 60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nSaved files:")
print("  - drowsiness_logistic_model.pkl")
print("  - scaler.pkl")
print("  - model_metrics.csv")
print("  - model_summary.txt")
print("  - eda_overview.png")
print("  - sample_images.png")
print("  - average_faces.png")
print("  - metrics_comparison.png")
print("  - confusion_matrices.png")
print("  - roc_curve.png")
print("  - precision_recall_curve.png")
print("  - prediction_distribution.png")