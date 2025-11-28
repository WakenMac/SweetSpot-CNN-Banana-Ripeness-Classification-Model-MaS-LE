import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_losses(subset):
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(
        subset['epoch'],
        subset['train_loss'],
        label='Train Loss',
        color='blue'
    )

    plt.plot(
        subset['epoch'],
        subset['validation_loss'],
        label='Validation Loss',
        color='orange'
    )

    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (CrossEntropy)')
    plt.legend()
    plt.grid(True)

def plot_accuracy(subset):
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(
        subset['epoch'],
        subset['train_accuracy'],
        label='Train Accuracy',
        color='blue'
    )

    plt.plot(
        subset['epoch'],
        subset['validation_accuracy'],
        label='Validation Accuracy',
        color='orange'
    )

    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

main_dataset = pd.read_csv('main_dataset.csv')
main_dataset['with_ReduceLROnPlateau'] = False
main_dataset['with_weight_decay'] = False
main_dataset.sort_values(['validation_accuracy', 'train_accuracy'], ascending=[False, False]).head(12)
main_dataset.sort_values(['train_accuracy', 'validation_accuracy'], ascending=[False, False]).head(12)

main_dataset.to_csv('main_dataset.csv', index=False)

subset = main_dataset[np.logical_and(main_dataset['batch_size'] == 64, main_dataset['learning_rate'] == 1e-03)]
plot_losses(subset)
plot_accuracy(subset)
