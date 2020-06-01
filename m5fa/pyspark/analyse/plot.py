from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure

print(matplotlib.get_backend())

fig: Figure = plt.figure()

x = [1, 2, 3, 2, 5, 7, 8, 9]
y = x

pbase = Path("/opt/data")
pplot = pbase / "plot"
if not pplot.exists():
    pplot.mkdir()


ax: Axis = fig.add_subplot(1, 1, 1)
ax.set_title("Empty title")
ax.plot(x, y)

pf1 = pplot / "f3.svg"
fig.savefig(pf1)
print(f"wrote to {pf1}")
