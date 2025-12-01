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
part1 = new_df.iloc[0:151, :]

# GMT 64
part2 = new_df.iloc[150:234, :]

# GMT 64 3e-05
part9 = new_df.iloc[234:285, :]

# GMT 64 1e-05 (Conti)
part10 = new_df.iloc[291:304, :]

# GMT 128
part3 = new_df.iloc[304:497, :]

# 64 3e-05
part4 = new_df.iloc[497:512, :]

# VGG19 64
part6 = new_df.iloc[534:618, :]

# VGG19 128
part7 = new_df.iloc[618:768, :]

# VGG19 32
part8 = new_df.iloc[770:, :]

new_pd = pd.concat([part1, part2, part9, part4, part10, part3, part8, part6, part7], axis=0, ignore_index = True)
new_pd.to_csv('temp_details.csv', index=False)

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

subset = main_dataset[main_dataset['model'] == 'VGG19']
subset['gap'] = subset['train_accuracy'] - subset['validation_accuracy']
subset.groupby(['model', 'batch_size', 'learning_rate']).last().reset_index().sort_values(['validation_accuracy', 'train_accuracy', 'gap'], ascending=[False, False, True]).head(20)
subset.groupby(['model', 'batch_size', 'learning_rate']).last().reset_index().sort_values(['train_accuracy', 'validation_accuracy', 'gap'], ascending=[False, False, True]).head(20)

subset['learning_rate'] = pd.to_numeric(subset['learning_rate'].astype(str).str.strip("'"), errors='coerce')
subset = subset[
    (subset['batch_size'] == 32) & 
    (subset['learning_rate'] == 1e-03)
]
subset = subset[np.logical_or(subset['epoch'] <= 50, np.logical_and(subset['epoch'] > 50, subset['with_weight_decay'] == True))]
plot_losses(subset)
plot_accuracy(subset)

other_df = pd.read_csv('training_details8.csv')
other_df['learning_rate'] = other_df['learning_rate'].apply(lambda x: 
    x.replace('[', '').replace(']', '').split(',')[-1]
)
other_df.drop_duplicates(inplace=True, ignore_index=True)

other_df.to_csv('training_details8.csv', index=False)
