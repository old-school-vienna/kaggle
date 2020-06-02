from pathlib import Path
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

def tupl(key: str) -> Tuple:
    return key, np.random.random(100)
    

keys = ['2000_01', "2000_02", "2000_03", "2000_04"]
ts = [tupl(k) for k in keys]

print(ts)

fig: Figure = plt.figure()

pbase = Path("/opt/data")
pplot = pbase / "plot"
if not pplot.exists():
    pplot.mkdir()

ax: Axes = fig.add_subplot(1, 1, 1)
print(f"--- type ax {type(ax)}")
ax.set_title("Boxes")
labs = [t[0] for t in ts]
vals = [t[1] for t in ts]
ax.boxplot(vals, labels=labs)

pf1 = pplot / "box1.svg"
fig.savefig(pf1)
print(f"wrote to {pf1}")
