import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── 1. Configuration ──────────────────────────────────────────────────────────
# Point this to the test folder and your friend's model weights
DATA_DIR = "dataset/fer2013_val/test"
CKPT_PATH = "models/best_fer_resnet34.pth"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating matrix using device: {device}")

    # ── 2. Data Loading ───────────────────────────────────────────────────────
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=DATA_DIR, transform=val_test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    class_names = test_dataset.classes
    print(f"Found {len(test_dataset)} images belonging to {len(class_names)} classes.")

    # ── 3. Model Definition (Torchvision ResNet-34) ───────────────────────────
    model = models.resnet34(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    
    # Load the best weights
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # ── 4. Inference Loop ─────────────────────────────────────────────────────
    all_preds = []
    all_labels = []

    print("Running Test Set through the model. Please wait...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ── 5. Calculate & Plot Confusion Matrix ──────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate overall accuracy for the title
    acc = np.trace(cm) / float(np.sum(cm))

    # Academic-Grade Plotting
    plt.figure(figsize=(10, 8), facecolor='#f4f6f9')
    ax = plt.axes()
    ax.set_facecolor('#f4f6f9')

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14, "weight": "bold"}) 

    # Formatting
    plt.title(f'Confusion Matrix: ResNet-34 on FER-2013\nOverall Accuracy: {acc*100:.2f}%', 
              fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Actual Emotion', fontsize=14)
    plt.xlabel('Predicted Emotion', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()

    # Save High-Resolution Image
    output_filename = 'FER2013_Confusion_Matrix.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ High-resolution Confusion Matrix successfully saved as '{output_filename}'!")

if __name__ == "__main__":
    main()