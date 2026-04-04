import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ── Model Definition ────────────────────────────────────────────────────────
class FERModel(nn.Module):
    def __init__(self, model_arch='resnet34', pretrained=False, num_classes=7):
        super().__init__()
        self.model = timm.create_model(
            model_arch, pretrained=pretrained, num_classes=num_classes, in_chans=3
        )
    def forward(self, x):
        return self.model(x)

# ── Configuration ───────────────────────────────────────────────────────────
# UPDATE THIS TO YOUR NEWEST MODEL AFTER TRAINING
CKPT_PATH = "models/resnet34-epoch.15-val_acc.68.26.ckpt" 

# Target the completely unseen TEST directory
DATA_DIR  = "dataset/fer2013/test"
CLASSES   = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 1. Load Data
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_data = datasets.ImageFolder(DATA_DIR, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

    # 2. Load Model
    model = FERModel(pretrained=False).to(device)
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("Model loaded successfully.")

    # 3. Run Inference
    y_true = []
    y_pred = []

    print("Running unseen test set predictions...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 4. Generate Academic Metrics
    print("\n" + "="*50)
    print("AFFECTIVE MODULE EVALUATION RESULTS (BLIND TEST SET)")
    print("="*50)
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    print(report)

    # 5. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8), facecolor='#f4f6f9')
    ax = plt.axes()
    ax.set_facecolor('#f4f6f9')
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES,
                annot_kws={"size": 14, "weight": "bold"})
    
    # Grab accuracy from report for title
    acc = np.trace(cm) / float(np.sum(cm))
    
    plt.title(f'Confusion Matrix — ResNet-34 FER2013\nUnseen Test Accuracy: {acc*100:.2f}%', 
              fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Emotion', fontsize=12)
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.tight_layout()
    
    plt.savefig("emotion_confusion_matrix_final.png", dpi=300, bbox_inches='tight')
    print("Saved high-resolution matrix to: emotion_confusion_matrix_final.png")

if __name__ == "__main__":
    main()