from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ds: pd.DataFrame = pd.read_csv("data/test-short.csv")
index: pd.Index = ds.keys()

pix = 28 * 28
print(f"28 * 28 = {pix}")

vals: np.ndarray = ds.values

imgs = vals[:, 0:]

print(f"vals shape: {vals.shape}")
print(f"img shape: {imgs.shape}")

imgs1: np.ndarray = imgs / 255.0


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int


def split_img(img: np.array) -> np.array:
    return np.asarray(np.split(img, 28))


imgs2 = np.apply_along_axis(split_img, 1, imgs1)

img = imgs2[2]

plt.imshow(img, cmap='Greys')
plt.show()
