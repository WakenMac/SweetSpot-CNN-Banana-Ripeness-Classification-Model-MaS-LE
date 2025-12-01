import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# For model evaulation
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

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

# Load pretrained VGG19
vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

# Freeze all VGG19 weights
for param in vgg19.parameters():
    param.requires_grad = False

class VGG19Transfer(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Pretrained VGG19 feature extractor
        self.features = vgg19.features

        # Global average pooling (like TF's GlobalAveragePooling2D)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ANN classifier block
        self.classifier = nn.Sequential(
            nn.Flatten(),               # flatten after GAP
            nn.Linear(512, 128),        # VGG19 outputs 512 feature maps
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes) # final layer (no softmax needed)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x  # raw logits; use CrossEntropyLoss

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        # patience: How many epochs to wait after the last best result
        self.patience = patience
        # min_delta: Minimum change to qualify as an improvement
        self.min_delta = min_delta

        self.counter = 0
        self.best_validation_loss = float('inf')
        self.should_stop = False

    def check_stop(self, validation_loss, learning_rate, batch_size):
        if validation_loss < self.best_validation_loss - self.min_delta:
            # Improvement found! Reset counter and update best loss
            self.best_validation_loss = validation_loss
            self.counter = 0
            # Save the current best model
            torch.save(model.state_dict(), f'Saved Models\\vgg_19_model_{batch_size}_{learning_rate}.pth')
            print("Validation loss improved. Model saved.")
        else:
            # No improvement
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

print('Starting model training...')

# Basic technique for training the model without validation accuracy
# for epoch in range(50):
    # model.train()
    # for images, labels in train_loader:
    #     images = images.to(device)
    #     labels = labels.to(device)

    #     optimizer.zero_grad()
    #     outputs = model(images)
    #     loss = criterion(outputs, labels)

    #     loss.backward()
    #     optimizer.step()

    # print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Things to consider: Batch sizes from 16 and 64 as well as  dropout rates
# reducing dropout (0.1–0.3) if underfitting
# increasing dropout (0.5–0.6) if overfitting

# Original
# learning_rates = [3e-05]

batch_size_list = [128]
learning_rates = [1e-03, 1e-04, 1e-05]

# learning_rates = [2e-03, 6e-04, 2e-04, 6e-05, 2e-05]
# learning_rates = [4e-03, 9e-04, 4e-04, 9e-05, 4e-05]
# learning_rates = [3e-04, 1e-04, 3e-05, 1e-05]

# One 64 model is reduced by 0.1 once, 43 epochs start

datasets = [train_ds, test_ds, val_ds]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

continue_value = 0

for batch_size in batch_size_list:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,            # like prefetching: loads data in parallel
        pin_memory=True           # speeds up GPU transfer
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    for i in range(len(learning_rates)):
        # For training from scratch
        model = VGG19Transfer(num_classes=4)

        # For fine-tuning
        if i == 0:
            model.load_state_dict(torch.load('Saved Models\\vgg_19_model_128_0.001.pth'))
        model.to(device)
        # model = torch.compile(model)
        torch.backends.cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()              # handles softmax internally
        optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rates[i],
            weight_decay=1e-4
        )

        early_stopper = EarlyStopper(patience=8, min_delta=1e-04)
        scheduler = ReduceLROnPlateau(
            optimizer = optimizer,
            mode='min',
            factor=0.1,
            patience=4,
            # min_lr=learning_rates[i] / 100
            min_lr=learning_rates[i] / 100
        )
        num_epochs = 50
        start = 1
        if i == 0:
            start = 16

        for epoch in range(start, num_epochs + 1):
            named_list = []
            accuracy_list = []
            loss_list = []
            validation_acc_list = []
            validation_loss_list = []
            epoch_list = []
            with_ReduceLROnPlateau = True
            with_weight_deacy = True

            model.train()
            # named_list.append(f'{learning_rates[i]}')
            # named_list.append(f'{str(learning_rates[i])}')
            named_list = str(learning_rates[i])
            epoch_list = epoch

            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Print every 10 batches
                if batch_idx % 10 == 0:
                    print(f'Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                        f'Loss: {running_loss / batch_idx:.4f}, '
                        f'Accuracy: {100 * correct / total:.2f}%')

            # Gets the epoch's overall accuracy and average loss
            accuracy_list.append(100 * correct / total)
            loss_list.append(running_loss / len(train_loader))

            # Validation after each epoch
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total

            # print(f'Learning Rate: {learning_rates[i]}, '
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}, '
                f'Epoch [{epoch}/{num_epochs}] completed. '
                f'Validation Loss: {avg_val_loss:.4f}, '
                f'Validation Accuracy: {val_accuracy:.2f}%\n')
            
            validation_acc_list.append(val_accuracy)
            validation_loss_list.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            # if early_stopper.check_stop(avg_val_loss, learning_rates[i], batch_sizes):
            if early_stopper.check_stop(avg_val_loss, learning_rates[i], batch_size):
                print(f"🛑 Early stopping triggered after {epoch} epochs!")
                break # Exit the training loop
        
            # Creating the template for training_details9.csv
            # pd.DataFrame(None, None, ['model', 'batch_size', 'learning_rate', 'epoch', 'train_accuracy', 'train_loss',
            #             'validation_accuracy', 'validation_loss', 'with_ReduceLROnPlateau',
            #             'with_weight_decay']).to_csv('training_details10.csv', index=False)

            df = pd.DataFrame({
                'model':'VGG19',
                'batch_size':batch_size,
                'learning_rate':str(named_list),
                'epoch':epoch_list,
                'train_accuracy':accuracy_list,
                'train_loss': loss_list,
                'validation_accuracy': validation_acc_list,
                'validation_loss':validation_loss_list,
                'with_ReduceLROnPlateau':with_ReduceLROnPlateau,
                'with_weight_decay':with_weight_deacy
            })

            pd.concat([pd.read_csv('training_details10.csv'), df], axis=0, ignore_index=True).to_csv('training_details10.csv', index=False)

# 3. Load the best model after training finishes
model = VGG19Transfer(num_classes=4).to(device)
# model.load_state_dict(torch.load('Saved Models\\best_gimatag_model_2_64_3e-05.pth'))
model.load_state_dict(torch.load('Saved Models\\vgg_19_model_32_0.001.pth'))
print("Loaded the best performing model from 'best_gimatag_model.pth'.")

# Print Model Output Parameters:
for index, p in enumerate(model.named_parameters()):
    print(f'Layer #{index + 1} ({p[0]}): {p[1].numel():,}')

# Prints the total number of parameters
print(f'Total Parameters: {sum([p.numel() for p in model.parameters()]):,}')

# Prediction
model.eval()
total_correct = 0
total_labels = 0
predicted_labels = []
all_labels = []

with torch.no_grad():
    print('Starting model evaluation now.')
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        _, predicted = output.max(1)

        all_labels.append(labels)
        predicted_labels.append(predicted)

        total_labels += labels.size(0)
        total_correct += predicted.eq(labels).sum().item()

test_accuracy = 100 * (total_correct / total_labels)
print(f"Test Accuracy: {total_correct}/{total_labels} ({test_accuracy:.2f}%)")

result = pd.DataFrame({
    'predicted':torch.cat(predicted_labels).cpu().numpy(),
    'truth':torch.cat(all_labels).cpu().numpy(),
})

cm = confusion_matrix(result['truth'], result['predicted'])
print(classification_report(result['truth'], result['predicted']))
print(cm)
plt.clf()
plot = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xticks([0.5, 1.5, 2.5, 3.5], ['Overripe', 'Ripe', 'Rotten', 'Unripe'])
plt.yticks([0.5, 1.5, 2.5, 3.5], ['Overripe', 'Ripe', 'Rotten', 'Unripe'])
