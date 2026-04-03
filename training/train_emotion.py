import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# 🧠 Model Definition
# =========================
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*4*4, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 📊 Evaluation Function
# =========================
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


# =========================
# 🚀 Main Training
# =========================
def main():
    # ✅ Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # 🎯 Transforms
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Step 1: Loading train data...")
    train_data = datasets.ImageFolder("dataset/fer2013/train", transform=transform)

    print("Step 2: Loading test data...")
    test_data = datasets.ImageFolder("dataset/fer2013/test", transform=transform)

    print("Step 3: Creating DataLoader...")
    train_loader = DataLoader(
        train_data,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print("Step 4: Starting training...")

    # 🧠 Model
    model = EmotionCNN().to(device)

    # ⚙️ Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 🔁 Training loop
    epochs = 15
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 📊 Evaluate
        acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

        # 💾 Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/best_emotion_model.pth")
            print("✅ Best model saved!")

    print("🎉 Training complete!")


# =========================
# 🔁 ONNX Export
# =========================
def export_onnx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load("models/best_emotion_model.pth"))
    model.eval()

    dummy_input = torch.randn(1, 1, 48, 48).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        "models/emotion_model.onnx",
        input_names=["input"],
        output_names=["output"]
    )

    print("✅ ONNX model exported!")


# =========================
# 🟢 Entry Point (IMPORTANT for Windows)
# =========================
if __name__ == "__main__":
    main()

    # Uncomment after training
    export_onnx()