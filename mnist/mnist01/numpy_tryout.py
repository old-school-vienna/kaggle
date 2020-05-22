import numpy as np


def thresh_x(x: np.array) -> (int, int):
    print(f"x:{x}")
    x_i = 0
    found = False
    for i, v in enumerate(np.nditer(x)):
        print(f"i v:{i} {v}")
        if v > 0.0 and not found:
            x_i = i
            found = True

    x_j = 0
    found = False
    y = np.flip(x)
    print(f"y:{y}")
    for i, v in enumerate(np.nditer(x[::-1])):
        print(f"i v:{i} {v}")
        if v > 0.0 and not found:
            x_j = i
            found = True
    return x_i, x_j


_x = np.array([0.0, 0.0, 0.5, 0.1, 0.2])
print(_x)
print(thresh_x(_x))
