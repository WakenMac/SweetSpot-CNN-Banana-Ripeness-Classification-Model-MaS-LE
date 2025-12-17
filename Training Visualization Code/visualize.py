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
main_dataset = main_dataset[np.logical_not(np.logical_and(main_dataset['model'] == 'VGG19', main_dataset['batch_size'] == 32))]
main_dataset = main_dataset[['model', 'batch_size', 'learning_rate', 'epoch', 'train_accuracy', 'train_loss',
'validation_accuracy', 'validation_loss', 'with_ReduceLROnPlateau',
'with_weight_decay']]
# main_dataset.to_csv('temp_details.csv', index=False)
main_dataset.sort_values(['validation_accuracy', 'train_accuracy'], ascending=[False, False]).head(12)
main_dataset.sort_values(['train_accuracy', 'validation_accuracy'], ascending=[False, False]).head(12)

subset = pd.concat([
    pd.read_csv('training_details11.csv'),
    pd.read_csv('training_details13(64_1e-05).csv'),
    pd.read_csv('training_details12(128).csv')
], axis=0, ignore_index = True)

pd.concat([main_dataset, subset], axis=0, ignore_index=True).to_csv('main_dataset.csv', index=False)

subset = pd.read_csv('training_details8.csv')
subset = subset.iloc[1:, :]
subset = subset.groupby(['model', 'batch_size', 'learning_rate', 'epoch'], sort=False).last().reset_index()
subset = subset.sort_values(['learning_rate'], ascending=[True])
subset.to_csv('temp_details.csv', index=False)

new_df = pd.concat([main_dataset, subset], axis=0, ignore_index = True)
new_df = new_df.groupby(['model', 'batch_size', 'learning_rate']).apply(lambda x: 
    x.sort_values('epoch', ascending=True)
).reset_index(drop=True)
new_df.to_csv('temp_details.csv', index=False)


# GMT 32
part1 = main_dataset.iloc[0:988, :]

# GMT 32 1e-04 fine tuning
part2 = pd.read_csv('training_details13.csv')

# GMT 32 1e-05 ... 128 3e-05
part3 = main_dataset.iloc[988:, :]


new_pd = pd.concat([
    part1, part2, part3
], axis=0, ignore_index = True)
new_pd.to_csv('main_dataset.csv', index=False)

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

pd.read_csv('temp_details.csv').to_csv('main_dataset.csv', index=False)

# For da plot


# Subsetting for the 
subset = main_dataset[main_dataset['model'] == 'GiMaTag']
subset['gap'] = subset['train_accuracy'] - subset['validation_accuracy']
# subset[subset['epoch'] <= 50] \
subset \
    .groupby(['model', 'batch_size', 'learning_rate']).last().reset_index() \
    .sort_values(['train_accuracy', 'validation_accuracy', 'gap'], ascending=[False, False, True]) \
    [['epoch', 'batch_size', 'learning_rate', 'train_accuracy', 'validation_accuracy', 'gap']] \
    .head(20)
subset.groupby(['model', 'batch_size', 'learning_rate']).last().reset_index().sort_values(['train_accuracy', 'validation_accuracy', 'gap'], ascending=[False, False, True]).head(20)

subset = main_dataset[main_dataset['model'] == 'ResNet50']
subset[subset['epoch'] == 50] \
    .groupby(['model', 'batch_size', 'learning_rate']).last().reset_index() \
    .value_counts()

subset = main_dataset[main_dataset['model'] == 'ResNet50']
subset['gap'] = subset['train_accuracy'] - subset['validation_accuracy']
subset['loss_gap'] = subset['train_loss'] - subset['validation_loss']
subset = subset[
    (subset['batch_size'] == 128) & 
    (subset['learning_rate'] == 3e-04)
    # (subset['epoch'] <= 50)
]
subset.iloc[-1, :]
subset = subset[np.logical_or(subset['epoch'] <= 50, np.logical_and(subset['epoch'] > 50, subset['with_weight_decay'] == True))]
plot_losses(subset)
plot_accuracy(subset)

other_df = pd.read_csv('training_details8.csv')
other_df['learning_rate'] = other_df['learning_rate'].apply(lambda x: 
    x.replace('[', '').replace(']', '').split(',')[-1]
)
other_df.drop_duplicates(inplace=True, ignore_index=True)

other_df.to_csv('training_details8.csv', index=False)
