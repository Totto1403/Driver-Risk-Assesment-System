import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os

# =============================================================================
# 🧠 Model Definition (Matches your analyzers.py exactly)
# =============================================================================
class FERModel(nn.Module):
    def __init__(self, model_arch: str = 'resnet34', pretrained: bool = True, num_classes: int = 7):
        super().__init__()
        # in_chans=3 because we convert 48x48 grayscale to 3-channel RGB for ResNet
        self.model = timm.create_model(
            model_arch, pretrained=pretrained, num_classes=num_classes, in_chans=3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# =============================================================================
# 📊 Evaluation Function
# =============================================================================
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total

# =============================================================================
# 🚀 Main Training Loop
# =============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Starting training on: {device}")

    # 1. Advanced Augmentation for Transfer Learning
    # ResNet expects 224x224 RGB images with ImageNet normalization
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # Convert to 3-channel
        transforms.Resize((224, 224)),               # Upscale for ResNet
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    train_data = datasets.ImageFolder("dataset/fer2013/train", transform=train_transform)
    test_data  = datasets.ImageFolder("dataset/fer2013/test", transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Load Pre-trained ResNet-34
    model = FERModel(pretrained=True).to(device)

    # 3. Handle Class Imbalance (Crucial for FER2013)
    # Approx distribution: [Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise]
    class_counts = [3995, 436, 4097, 7215, 4965, 4830, 3171] 
    total_samples = sum(class_counts)
    class_weights = [total_samples / c for c in class_counts]
    weights_tensor = torch.FloatTensor(class_weights).to(device)

    # Apply the weights to the loss function
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    # 4. Optimizer & Learning Rate Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    epochs = 20
    best_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # Evaluate at end of epoch
        acc = evaluate(model, test_loader, device)
        avg_loss = total_loss / len(train_loader)
        
        print(f"====== Epoch {epoch+1} Summary ======")
        print(f"Avg Loss: {avg_loss:.4f} | Val Accuracy: {acc:.2f}%")
        
        # Step the scheduler based on validation accuracy
        scheduler.step(acc)

        # 5. Save in the EXACT format expected by main_integrated.py
        if acc > best_acc:
            best_acc = acc
            ckpt_path = f"models/resnet34-epoch.{epoch+1}-val_acc.{acc:.2f}.ckpt"
            
            # Pack state dict to match your inference code cp["state_dict"]
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc": best_acc
            }
            torch.save(checkpoint, ckpt_path)
            print(f"🔥 NEW BEST MODEL SAVED: {ckpt_path}")

    print(f"🎉 Training complete! Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()