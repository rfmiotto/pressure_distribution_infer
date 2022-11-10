"""
Customize colormap
"""
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap


def trans_color(color, alpha=0.4):
    return matplotlib.colors.colorConverter.to_rgba(color, alpha=alpha)


def transparent_cmap(color="#63599e", alpha=0.4, N=512):
    c_invisible = matplotlib.colors.colorConverter.to_rgba("white", alpha=0)
    c_white = matplotlib.colors.colorConverter.to_rgba("white", alpha=alpha)
    c_custom = matplotlib.colors.colorConverter.to_rgba(color, alpha=alpha)
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom_cmap", [(0.0, c_invisible), (0.05, c_white), (1.0, c_custom)], N=N
    )


def transparent_cmap2(colorCold="#4a6b3d", colorHot="#753b6f", alpha=0.4, N=512):
    c_invisible = matplotlib.colors.colorConverter.to_rgba("white", alpha=0)
    c_white = matplotlib.colors.colorConverter.to_rgba("white", alpha=alpha)
    c_cold = matplotlib.colors.colorConverter.to_rgba(colorCold, alpha=alpha)
    c_hot = matplotlib.colors.colorConverter.to_rgba(colorHot, alpha=alpha)
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom_cmap",
        [
            (0.0, c_cold),
            (0.49, c_white),
            (0.5, c_invisible),
            (0.51, c_white),
            (1.0, c_hot),
        ],
        N=N,
    )


def custom_bwr(N=512, reverse=False):
    color1 = "royalblue"
    color2 = "firebrick"
    if reverse:
        color1 = "firebrick"
        color2 = "royalblue"
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "customBWR", [(0, color1), (0.5, "white"), (1, color2)], N=N
    )


def custom_orange_blue(N=512, reverse=False):
    num = int(N / 2)
    top = cm.get_cmap("Blues_r", num)
    bottom = cm.get_cmap("Oranges", num)
    newcolors = np.vstack((top(np.linspace(0, 1, num)), bottom(np.linspace(0, 1, num))))
    if reverse:
        return ListedColormap(newcolors[::-1], name="OrangeBlue")
    return ListedColormap(newcolors, name="OrangeBlue")


def custom_magma(N=512, reverse=False):
    magma = cm.get_cmap("magma", N)
    perc = 0.93  # percentage beyond which the color will change
    N_magma = int(N * perc)
    N_white = N - N_magma
    # newcolors = magma(np.linspace(0, perc, ))
    bottom = cm.get_cmap("magma", N_magma)
    top = matplotlib.colors.LinearSegmentedColormap.from_list(
        "white_end", [(0.0, magma(1.0)), (1.0, "white")], N=N_white
    )
    newcolors = np.vstack(
        (bottom(np.linspace(0, 1, N_magma)), top(np.linspace(0, 1, N_white)))
    )
    if reverse:
        return ListedColormap(newcolors[::-1], name="customMagma")
    return ListedColormap(newcolors, name="customMagma")


def custom_gnbu(N=512, reverse=False):
    GnBu = cm.get_cmap("GnBu_r", N)
    perc = 0.93  # percentage beyond which the color will change
    N_GnBu = int(N * perc)
    N_white = N - N_GnBu
    # newcolors = GnBu(np.linspace(0, perc, ))
    bottom = cm.get_cmap("GnBu_r", N_GnBu)
    top = matplotlib.colors.LinearSegmentedColormap.from_list(
        "white_end", [(0.0, GnBu(1.0)), (1.0, "white")], N=N_white
    )
    newcolors = np.vstack(
        (
            bottom(np.linspace(0, 1, N_GnBu)),
            top(np.linspace(0, 1, N_white)),
        )
    )
    if reverse:
        return ListedColormap(newcolors[::-1], name="customGnBu")
    return ListedColormap(newcolors, name="customGnBu")
