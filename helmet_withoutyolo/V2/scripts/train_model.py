import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader

# =========================
# PATHS
# =========================
train_dir = r"D:\Dev\Coding\Safesight\helmet_withoutyolo\V2\data\final\train"
val_dir = r"D:\Dev\Coding\Safesight\helmet_withoutyolo\V2\data\final\val"

# =========================
# TRANSFORMS (FIXED 🔥)
# =========================
train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# DATA
# =========================
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

print("Classes:", train_data.classes)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# MODEL (UPDATED API 🔥)
# =========================
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)

# Freeze feature extractor (IMPORTANT)
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, 2)

model = model.to(device)

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0005)

# =========================
# PERFORMANCE BOOST
# =========================
torch.backends.cudnn.benchmark = True

# =========================
# TRAINING LOOP
# =========================
epochs = 15   # 🔥 don't overtrain

best_acc = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # =========================
    # VALIDATION
    # =========================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total

    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Val Acc: {acc:.4f}")

    # Save best models
    if acc > best_acc:
        best_acc = acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': train_data.classes
        }, "helmet_model_best.pth")

print("✅ Training complete")
print("Best Accuracy:", best_acc)