import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# =========================
# 🧠 Model Definition (Improved)
# =========================
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 📊 Evaluation
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
# 💾 Load previous best accuracy
# =========================
def load_best_accuracy(filepath="models/best_accuracy.txt"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return float(f.read().strip())
    return 0.0


def save_best_accuracy(acc, filepath="models/best_accuracy.txt"):
    with open(filepath, "w") as f:
        f.write(str(acc))


# =========================
# 🚀 Training
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # 🔥 Improved augmentation
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Loading datasets...")
    train_data = datasets.ImageFolder("dataset/fer2013/train", transform=transform)
    test_data  = datasets.ImageFolder("dataset/fer2013/test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    model = EmotionCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    epochs = 25

    # 🔥 Load previous best
    best_acc = load_best_accuracy()
    print(f"Previous best accuracy: {best_acc:.2f}%")

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

        acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

        # 💾 Save only if better than ALL previous runs
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/best_emotion_model.pth")
            save_best_accuracy(best_acc)
            print("🔥 New BEST model saved across runs!")

    print("🎉 Training complete!")


# =========================
# 🔁 ONNX Export
# =========================
def export_onnx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load("models/best_emotion_model.pth", weights_only=True))
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
# 🟢 Entry
# =========================
if __name__ == "__main__":
    main()
    # export_onnx()