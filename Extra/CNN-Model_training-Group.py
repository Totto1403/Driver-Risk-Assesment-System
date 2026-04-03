import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ============================================
# CRITICAL FIX: Force TensorFlow to use GPU
# ============================================

print("="*60)
print("DIAGNOSING GPU ISSUE")
print("="*60)

# Check TensorFlow build
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# List all devices
print("\n--- Available Devices ---")
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Check GPU devices
physical_devices = tf.config.list_physical_devices('GPU')
print(f"\n--- Physical GPU Devices ---")
print(f"Number of GPUs: {len(physical_devices)}")
for device in physical_devices:
    print(f"  {device}")

# Check if GPU is actually available for TensorFlow
print(f"\nGPU Available (deprecated check): {tf.test.is_gpu_available()}")

# CRITICAL: Check CUDA and cuDNN versions
print("\n--- CUDA Configuration ---")
print(f"CUDA Version: {tf.sysconfig.get_build_info()['cuda_version']}")
print(f"cuDNN Version: {tf.sysconfig.get_build_info()['cudnn_version']}")

print("="*60 + "\n")

# ============================================
# THE REAL PROBLEM: Check if TensorFlow can see GPU
# ============================================

if len(physical_devices) == 0:
    print("❌ PROBLEM FOUND: TensorFlow cannot detect GPU!")
    print("\nSOLUTIONS:")
    print("1. In Kaggle notebook settings (right panel):")
    print("   - Click 'Session Options' or the three dots menu")
    print("   - Under 'Accelerator', select 'GPU T4 x2'")
    print("   - Click 'Restart & Run All'")
    print("\n2. Make sure you're using a TensorFlow version with GPU support")
    print("   (Kaggle usually has this by default)")
    print("\n3. Try restarting the kernel completely")
    print("="*60)
    raise SystemExit("GPU not detected - please enable GPU in Kaggle settings")

# Configure GPU properly
print("✓ GPU detected! Configuring...")
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
print("✓ Memory growth enabled")

# ============================================
# TEST: Verify GPU is actually being used
# ============================================

print("\n" + "="*60)
print("TESTING GPU USAGE")
print("="*60)

# Create a simple computation and check which device it runs on
print("\nRunning test computation...")
with tf.device('/GPU:0'):
    # Create random tensors
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    
    # Do matrix multiplication (GPU-intensive operation)
    import time
    start = time.time()
    c = tf.matmul(a, b)
    c.numpy()  # Force execution
    gpu_time = time.time() - start
    print(f"✓ GPU computation time: {gpu_time:.4f} seconds")

# Compare with CPU
print("\nRunning same test on CPU...")
with tf.device('/CPU:0'):
    a_cpu = tf.random.normal([10000, 10000])
    b_cpu = tf.random.normal([10000, 10000])
    
    start = time.time()
    c_cpu = tf.matmul(a_cpu, b_cpu)
    c_cpu.numpy()
    cpu_time = time.time() - start
    print(f"✓ CPU computation time: {cpu_time:.4f} seconds")

print(f"\n{'='*60}")
if gpu_time < cpu_time:
    speedup = cpu_time / gpu_time
    print(f"✓ GPU is {speedup:.1f}x FASTER - GPU IS WORKING!")
else:
    print("⚠ WARNING: GPU is not faster than CPU - something is wrong!")
print(f"{'='*60}\n")

# ============================================
# DATA LOADING - TRAIN AND TEST
# ============================================


# ============================================
# DATA LOADING - TRAIN AND TEST (FIXED)
# ============================================

print("="*60)
print("DATA LOADING (FIXED TO USE VALIDATION_SPLIT)")
print("="*60)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = "/kaggle/input/nthuddd2"
TRAIN_VAL_DIR = f"{DATASET_DIR}/train_data" # Renamed to reflect it holds all data
# We no longer need the non-existent TEST_DIR variable

# Define the split percentage
VALIDATION_SPLIT = 0.2 # 20% of data will be used for testing/validation
SEED = 42

# --- 1. Load the Training Dataset ---
print(f"Loading training data from: {TRAIN_VAL_DIR} (using {1-VALIDATION_SPLIT} for training)")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_VAL_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=VALIDATION_SPLIT, # IMPORTANT: Define the split percentage
    subset="training"                 # IMPORTANT: Specify this is the training subset
)

class_names = train_ds.class_names
print(f"✓ Training classes: {class_names}")
print(f"✓ Number of training batches: {len(train_ds)}")

# --- 2. Load the Testing/Validation Dataset ---
print(f"\nLoading test data (validation subset) from: {TRAIN_VAL_DIR} (using {VALIDATION_SPLIT} for testing)")

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_VAL_DIR,                       # IMPORTANT: Use the same directory
    labels='inferred',
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,                       # Don't shuffle test data
    seed=SEED,                           # IMPORTANT: Use the same seed
    validation_split=VALIDATION_SPLIT,   # IMPORTANT: Use the same split percentage
    subset="validation"                  # IMPORTANT: Specify this is the validation (test) subset
)

print(f"✓ Test classes: {test_ds.class_names}")
print(f"✓ Number of test batches: {len(test_ds)}")

# Optimize pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

print("="*60 + "\n")

# ============================================
# MODEL - EXPLICITLY ON GPU
# ============================================

print("="*60)
print("MODEL CREATION")
print("="*60)

# FORCE model creation on GPU
print("Creating model on /GPU:0...")
with tf.device('/GPU:0'):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])

print("✓ Model created")

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print("="*60 + "\n")

# ============================================
# TRAINING - FORCE GPU USAGE
# ============================================

print("="*60)
print("TRAINING ON GPU")
print("="*60)

EPOCHS = 10

# EXPLICITLY train on GPU
print(f"Training for {EPOCHS} epochs on /GPU:0...")
print("Watch the GPU usage in Kaggle's monitoring panel!\n")

# Log device placement to verify GPU usage
tf.debugging.set_log_device_placement(True)

with tf.device('/GPU:0'):
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        verbose=1,
        validation_data=test_ds  # Added validation during training
    )

print("\n✓ Training complete!")
print("="*60 + "\n")

# ============================================
# TESTING AND EVALUATION
# ============================================

print("="*60)
print("MODEL EVALUATION ON TEST SET")
print("="*60)

# Evaluate on test set
print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print(f"\n{'='*60}")
print(f"TEST RESULTS:")
print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Test Loss: {test_loss:.4f}")
print(f"{'='*60}\n")

# ============================================
# DETAILED PREDICTIONS AND METRICS
# ============================================

print("="*60)
print("GENERATING PREDICTIONS AND METRICS")
print("="*60)

# Get predictions
print("\nGenerating predictions...")
y_pred = model.predict(test_ds, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true labels
y_true = []
for images, labels in test_ds:
    y_true.extend(np.argmax(labels.numpy(), axis=1))
y_true = np.array(y_true)

# Classification report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Confusion matrix
print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)
cm = confusion_matrix(y_true, y_pred_classes)
print(cm)

# ============================================
# VERIFY GPU WAS USED
# ============================================

print("\n" + "="*60)
print("GPU USAGE VERIFICATION")
print("="*60)

os.system('nvidia-smi')

print("\n" + "="*60)
print("TRAINING AND TESTING COMPLETE")
print("="*60)

# ============================================
# VISUALIZATIONS
# ============================================

import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], marker='o', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], marker='s', label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], marker='o', color='red', label='Training Loss')
plt.plot(history.history['val_loss'], marker='s', color='orange', label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Confusion matrix heatmap
plt.subplot(1, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('training_testing_results.png', dpi=100)
plt.show()

# Save model
model.save('model.keras')
print("\n✓ Model saved as 'model.keras'")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Training Results:")
print(f"  Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"\nValidation Results:")
print(f"  Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"  Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"\nTest Results:")
print(f"  Test Accuracy: {test_accuracy:.4f}")
print(f"  Test Loss: {test_loss:.4f}")
print("\n" + "="*60)

# Per-class accuracy
print("\nPer-Class Accuracy:")
for i, class_name in enumerate(class_names):
    class_mask = y_true == i
    class_acc = np.sum(y_pred_classes[class_mask] == y_true[class_mask]) / np.sum(class_mask)
    print(f"  {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")

print("\n" + "="*60)
print("\nIf GPU usage was still low during training:")
print("1. Check Kaggle's GPU monitoring panel during training")
print("2. Your dataset might be too small - GPU benefits are minimal for small datasets")
print("3. Data loading might be the bottleneck, not computation")
print("="*60)