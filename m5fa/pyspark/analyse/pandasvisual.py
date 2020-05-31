from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure

fig: Figure = plt.figure()

x = [1, 2, 3, 4, 5]
y = x

#pbase = Path("/opt/data")
pbase = Path("/data/kaggle")
pplot = pbase / "plot"
if not pplot.exists():
    pplot.mkdir()

pf1 = pplot / "f3.png"
fig.savefig(pf1)

ax: Axis = fig.add_subplot(1, 1, 1)
ax.set_title("Empty title")
ax.plot(x, y)

plt.show()
print(f"wrote to {pf1}")
