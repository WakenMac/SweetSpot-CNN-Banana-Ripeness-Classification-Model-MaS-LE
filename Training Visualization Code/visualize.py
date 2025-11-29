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
main_dataset = main_dataset[['model', 'batch_size', 'learning_rate', 'epoch', 'train_accuracy', 'train_loss',
'validation_accuracy', 'validation_loss', 'with_ReduceLROnPlateau',
'with_weight_decay']]
main_dataset.sort_values(['validation_accuracy', 'train_accuracy'], ascending=[False, False]).head(12)
main_dataset.sort_values(['train_accuracy', 'validation_accuracy'], ascending=[False, False]).head(12)

subset = pd.read_csv('training_details9(64).csv')
subset = subset.iloc[1:, :]
subset = subset.groupby(['model', 'batch_size', 'learning_rate', 'epoch']).last().reset_index()
subset.to_csv('temp_details.csv')

subset1 = pd.concat([subset.iloc[1:51, :], subset.iloc[51:101, :], subset.iloc[151:201, :]], axis=0, ignore_index = True)

subset['model'] = 'GiMaTag'
subset = subset[['model', 'batch_size', 'learning_rate', 'epoch', 'train_accuracy', 'train_loss',
'validation_accuracy', 'validation_loss', 'with_ReduceLROnPlateau',
'with_weight_decay']]
main_dataset = pd.concat([main_dataset, subset], axis=0, ignore_index=True)
main_dataset['learning_rate'] = main_dataset['learning_rate'].apply(lambda x: str(x))
main_dataset.iloc[634:, 1] = 32
main_dataset.iloc[634:, 2] = str(float(1e-05))

main_dataset.to_csv('temp_details.csv', index=False)
main_dataset.to_csv('main_dataset.csv', index=False)

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
