from torch.utils.data import Dataset
from PIL import Image

class PneumoniaMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label
