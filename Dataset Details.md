## Datasets Utilized

### \[1\] GiMaTag Banana Dataset

The split version of the "Final Integrated Dataset", where the dataset is split into 80:10:10 ratio for training, validating, and testing the model.

**Composition (Images):**

|   | Unripe | Ripe | Overripe | Rotten | Total |
|------------|:----------:|:----------:|:----------:|:----------:|:----------:|
| **Train** | 2,575 (19.82%) | 4,300 (33.09%) | 2,416 (18.59%) | 3,674 (28.27%) | 12,995 (80%) |
| **Test** | 323 (19.89%) | 538 (33.13%) | 303 (18.66%) | 460 (28.33%) | 1,624 (10%) |
| **Validation** | 321 (19.82%) | 537 (33.17%) | 302 (18.65%) | 459 (28.35%) | 1,619 (10%) |
| **Total** | 3,219 (19.86%) | 5,375 (33.16%) | 3,021 (18.64%) | 4,593 (28.34%) | 16,208 (100%) |

### \[2\] Final Integrated Dataset

The dataset to be used for train, test, validation split.

This dataset is the combination of the two derived datasets:

1.  Shahriar Musa Acuminata Dataset
2.  Fayoum University Dataset (Annotated)

Details on the datasets mentioned can be found below.

**Composition (Images):**

|           |      Plantain      |   Musa Acuminata   |   Total   | Overall %  |
|:----------|:------------------:|:------------------:|:---------:|:----------:|
| Unripe    |   1,040 (32.31%)   |   2,179 (67.69%)   |   3,219   |   19.86%   |
| Ripe      | **1,360 (25.30%)** | **4,015 (74.70%)** | **5,375** | **33.16%** |
| Overripe  |    330 (10.92%)    |   2,691 (89.08%)   |   3,021   |   18.64%   |
| Rotten    |     0 (0.00 %)     |  4,593 (100.00 %)  |   4,593   |   28.34%   |
| **TOTAL** |       16.84%       |       83.16%       |  16,208   |    100%    |

Given the ratio of our dataset, the model is expected to predict ripe bananas, as well as Musa Acuminata bananas more accurately.

## Derrived Datasets

### \[3\] Shahriar Musa Acuminata Dataset

The dataset was taken from Kaggle: [Ripeness Classification Dataset (S.M. Shahriar)](https://www.kaggle.com/datasets/shahriar26s/banana-ripeness-classification-dataset). Similarly, details on the augmentations made are also listed in the link mentioned.

We took the images from the train, test, and validation folders and compiled them based on their class. The composition of the images are listed below:

**Composition (Images):**

|          | Number of Images |     \%     |
|----------|:----------------:|:----------:|
| Unripe   |      2,179       |   15.96%   |
| Ripe     |      4,015       |   29.42%   |
| Overripe |      2,691       |   19.72%   |
| Rotten   |    **4,593**     | **33.66%** |
| TOTAL    |      13,648      |    100%    |

**Re-classification made by the dataset:**

(New Class: Old Classes)

Unripe: Fresh unripe, unripe

Ripe: Banana, Fresh ripe, ripe

Overripe: Mold, Overripe

Rotten: Rotten, Unripe, Banana, Ripe

**Overall original classes:**

  Fresh unripe, unripe, fresh ripe, ripe, overripe, mold, rotten

### \[4\] Fayoum University Banana Classes

This dataset was utilized in several studies such as:

> \[5\]	N. Saranya, K. Srinivasan, and S. K. Pravin Kumar. 2021. Banana ripeness stage identification: a deep learning approach. Journal of Ambient Intelligence and Humanized Computing 13, 8: 4033–4039.
>
> \[6\]	Fatma M. A. Mazen and Ahmed A. Nashat. 2019. Ripeness classification of bananas using an artificial neural network. Arabian Journal for Science and Engineering 44, 8: 6901–6910.

It is accessible through this drive link: [Fayoum University Banana Classes](https://drive.google.com/drive/folders/1nRWBYAHNRqmL4R0SLrs6dbGQFSWGVY8V), and consists of 273 images of Plantain bananas with 4 classes as listed below.

**Composition (Images):**

|                                 | Number of Images |     \%     |
|---------------------------------|:----------------:|:----------:|
| Green (Unripe and Fresh Unripe) |     **104**      | **38.95%** |
| Midripen (Ripe)                 |        88        |   32.23%   |
| Overripen (Overripe)            |        33        |   12.09%   |
| Yellowish Green (Ripe)          |        48        |   17.58%   |
| TOTAL                           |       273        |    100%    |

### \[5\] Fayoum University Banana Classes (Augmented by Researchers)

It is the augmented version of the original dataset \[4\].

**Composition (Images):**

|          | Number of Images |     \%     |
|----------|:----------------:|:----------:|
| Unripe   |      1,040       | **38.95%** |
| Ripe     |      1,360       |   49.81%   |
| Overripe |       330        |   12.09%   |
| TOTAL    |      2,730       |    100%    |

**Augmentations Made:**

The augmentations made to the original images are from a range of following values:

1.  Gaussian Noise (0 to 50)
2.  Image Blur (0 or 1)
3.  Rotate Image (-30 to 30)
4.  Crop & Zoom (0 to 20)
5.  Flip Image (-1, 0, 1)\
    -1 = Flips image vertically and horizontally\
    0 = Flips image vertically\
    1 = Flips image horizontally

These augmentations result in 10 augmented images derived from 1 original image, leading to a 10x increase from the original 273 images of the original dataset. Additionally, the specific augmentations made are listed in the "Plaintain_Augment_List.csv" file, with the code to run the augmentations found in "Augment.py" file, where both files are found within this repository.

To find the locations of the files mentioned above, check the File Structure section the README.md of this repository.

**Re-classification made to the dataset:**

(New Class: Old Classes)

Unripe: Green

Ripe: Midripen, Yellowish Green

Overripe: Overripen