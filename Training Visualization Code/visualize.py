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
main_dataset['model'] = 'GiMaTag'
main_dataset = main_dataset[['model', 'batch_size', 'learning_rate', 'epoch', 'train_accuracy', 'train_loss',
'validation_accuracy', 'validation_loss', 'with_ReduceLROnPlateau',
'with_weight_decay']]
main_dataset.sort_values(['validation_accuracy', 'train_accuracy'], ascending=[False, False]).head(12)
main_dataset.sort_values(['train_accuracy', 'validation_accuracy'], ascending=[False, False]).head(12)

subset = pd.read_csv('training_details7.csv')
subset['model'] = 'GiMaTag'
subset['learning_rate'] = subset['named_list']
subset['with_weight_decay'] = subset['with_weight_deacay']
subset.iloc[:, 1] = 3e-05
subset = subset[['model', 'batch_size', 'learning_rate', 'epoch', 'train_accuracy', 'train_loss',
'validation_accuracy', 'validation_loss', 'with_ReduceLROnPlateau',
'with_weight_decay']]
main_dataset = pd.concat([main_dataset, subset], axis=0, ignore_index=True)

# main_dataset.to_csv('main_dataset.csv', index=False)

subset = main_dataset[np.logical_and(main_dataset['batch_size'] == 64, main_dataset['learning_rate'] == 3e-05)]
subset = subset[np.logical_or(subset['epoch'] <= 50, np.logical_and(subset['epoch'] > 50, subset['with_weight_decay'] == True))]
plot_losses(subset)
plot_accuracy(subset)

other_df = pd.read_csv('training_details8.csv')
other_df['learning_rate'] = other_df['learning_rate'].apply(lambda x: 
    x.replace('[', '').replace(']', '').split(',')[-1]
)
other_df.drop_duplicates(inplace=True, ignore_index=True)

other_df.to_csv('training_details8.csv', index=False)
