import matplotlib.pyplot as plt
import numpy as np


def show_digit_from_df(data, number_of_picture):
    image = data.loc[number_of_picture]
    image = image.drop("label")
    image = np.reshape(image.as_matrix(), (-1, 28))
    print(image)
    show_digit(image)


def show_digit(image):
    plt.imshow(image, cmap=plt.cm.gray, interpolation="nearest")
    plt.show()
