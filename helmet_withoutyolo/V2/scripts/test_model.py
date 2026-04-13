import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

# Load models
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)

checkpoint = torch.load("helmet_model_best.pth", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class_names = checkpoint['class_names']

# 🔥 IMPORTANT NORMALIZATION
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img_path = "img_1.png"
img = Image.open(img_path).convert("RGB")

input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    _, pred = torch.max(output, 1)

print("Prediction:", class_names[pred.item()])