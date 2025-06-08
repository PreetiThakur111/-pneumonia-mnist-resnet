import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from dataset import PneumoniaMNISTDataset

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    # Load validation data
    val_images = np.load("val_images.npy", allow_pickle=True)
    val_labels = np.load("val_labels.npy")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_dataset = PneumoniaMNISTDataset(images=val_images, labels=val_labels, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)

    evaluate_model(model, val_loader, device)