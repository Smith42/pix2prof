import matplotlib as mpl
mpl.use("Agg")

import numpy as np
from glob import glob
from os.path import basename
import matplotlib.pyplot as plt
from math import ceil

def plot_validation_set(rootdir):
    fis = list(map(basename, sorted(glob("{}/*-y.txt".format(rootdir)))))
    p_fis = sorted(glob("{}/*-p.txt".format(rootdir)))
    y_fis = sorted(glob("{}/*-y.txt".format(rootdir)))

    ps = map(np.loadtxt, p_fis)
    ys = map(np.loadtxt, y_fis)

    for batch in range(ceil(len(fis) / 36)):
        f, axs = plt.subplots(6, 6, sharey=True, sharex=True, figsize=(20, 20))
        for fi, p, y, ax in zip(fis, ps, ys, axs.ravel()):
            ax.set_title(fi[:-6])
            ax.plot(y, label="Target")
            ax.plot(p, label="Prediction")

        axs[0, 0].legend()
        plt.tight_layout()
        plt.savefig("{}/profiles-{}.png".format(rootdir, batch))
        plt.close()
