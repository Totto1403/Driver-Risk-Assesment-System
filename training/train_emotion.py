import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os

# =============================================================================
# 🧠 Model Definition
# =============================================================================
class FERModel(nn.Module):
    def __init__(self, model_arch: str = 'resnet34', pretrained: bool = True, num_classes: int = 7):
        super().__init__()
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

    # 1. Advanced Augmentation
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    # UPDATED: Pointing to train and val directories
    train_data = datasets.ImageFolder("dataset/fer2013_val/train", transform=train_transform)
    val_data   = datasets.ImageFolder("dataset/fer2013_val/val", transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Load Pre-trained ResNet-34
    model = FERModel(pretrained=True).to(device)

    # 3. Handle Class Imbalance (Recalculated roughly for the new 80% split)
    class_counts = [3196, 349, 3278, 5772, 3972, 3864, 2537] 
    total_samples = sum(class_counts)
    class_weights = [total_samples / c for c in class_counts]
    weights_tensor = torch.FloatTensor(class_weights).to(device)

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

        # Evaluate at end of epoch using the VALIDATION set
        acc = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        
        print(f"====== Epoch {epoch+1} Summary ======")
        print(f"Avg Loss: {avg_loss:.4f} | Validation Accuracy: {acc:.2f}%")
        
        # Step the scheduler based on validation accuracy
        scheduler.step(acc)

        # 5. Save model
        if acc > best_acc:
            best_acc = acc
            ckpt_path = f"models/resnet34-epoch.{epoch+1}-val_acc.{acc:.2f}.ckpt"
            
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc": best_acc
            }
            torch.save(checkpoint, ckpt_path)
            print(f"🔥 NEW BEST MODEL SAVED: {ckpt_path}")

    print(f"🎉 Training complete! Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()