import matplotlib.pyplot as plt
import numpy as np

from fast_image_classification.preprocessing_utilities import read_img_from_path
from fast_image_classification.training_utilities import get_seq


def plot_figures(names, figures, nrows=1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(128, 128))
    for ind, title in enumerate(names):
        img = np.squeeze(figures[title])
        if len(img.shape) == 2:
            axeslist.ravel()[ind].imshow(img, cmap=plt.gray())  # , cmap=plt.gray()
        else:
            axeslist.ravel()[ind].imshow(img)

        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()  # optional

    plt.show()


figures = {}
img_1 = read_img_from_path('../example/data/0d539aa7c5f448c5aca2db15eeebb0c3.jpg')
figures['Original'] = img_1

seq = get_seq()

for i in range(1, 12):
    augmented = seq.augment_image(img_1)
    figures["Augmented %s" % i] = augmented

names = ['Original'] + ["Augmented %s" % i for i in range(1, 10)]

# plot of the images in a figure, with 2 rows and 5 columns
plot_figures(names, figures, 2, 5)
