import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),                    # Converts to [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],           # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


dataset_path = "C:\\Users\\Waks\\Downloads\\USEP BSCS\\School Work\\BSCS 3 - 1st Sem\\CS 3310 - Modeling and Simulation\\MaS LE\\Datasets\\GiMaTag Dataset"

train_ds = datasets.ImageFolder(dataset_path + '/train', transform=train_transforms)
val_ds   = datasets.ImageFolder(dataset_path + '/validation', transform=test_transforms)
test_ds  = datasets.ImageFolder(dataset_path + '/test', transform=test_transforms)

print("Successfully imported the train, test, and validation datasets")

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,            # like prefetching: loads data in parallel
    pin_memory=True           # speeds up GPU transfer
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


class GiMaTagCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(GiMaTagCNN, self).__init__()
        
        # --- Block 1 ---
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )

        # --- Block 2 ---
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )

        # --- Block 3 ---
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.40)
        )

        # After 3 MaxPools from 224×224:
        # 224 → 112 → 56 → 28
        flattened_size = 64 * 28 * 28

        # --- Fully Connected (ANN) ---
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(0.50),

            nn.Linear(64, num_classes)
            # No softmax — PyTorch's CrossEntropyLoss expects raw logits
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = torch.flatten(x, 1)  # same as keras Flatten()
        x = self.fc(x)
        return x   # raw logits for CrossEntropyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GiMaTagCNN(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()              # handles softmax internally
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")



