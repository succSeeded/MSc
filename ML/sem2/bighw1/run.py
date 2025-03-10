"""
A blackbox optimizer that uses GP to approximate the function and finds the maximum of its mean. 
"""

from os.path import isfile, join
from time import sleep
from typing import Dict, Any
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from requests import post
from requests.exceptions import ReadTimeout
from matplotlib.pyplot import subplots, show
from matplotlib import colormaps
import numpy as np


def get_grid(x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """
    Get function values on the grid using POST requests.
    """
    post_args = {"secret_key": "WVRjjVBm", "type": "small"}
    x_len, y_len = x_coords.shape
    z = np.zeros((x_len, y_len))

    with open("fvals.txt", "a", encoding="utf8") as file:
        for i in range(x_len):
            for j in range(y_len):
                coords = str(x_coords[i,j]) + " " + str(y_coords[i,j])
                print(f"Requesting value at {coords}\n")

                post_args["x"] = coords

                try:
                    response = post("http://optimize-me.ddns.net:8080/", data=post_args, timeout=2)
                except ReadTimeout:
                    response = post("http://optimize-me.ddns.net:8080/", data=post_args, timeout=2)

                fval = str(response.content, encoding="utf8")
                z[i,j] = np.float64(response.content)
                file.write(coords + "|" + fval + "\n")

                sleep(1)
    return z


if __name__=="__main__":

    nx = 11
    ny = 11
    x, y = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))

    if not isfile(join("E:\\Work\\Studies\\MSc\\ML\\sem2\\bighw1", ".\\fvals.txt")):
        z = get_grid(x, y)
    else:
        with open("fvals.txt", "r") as file:
            i = 0
            z = np.zeros(nx*ny)
            for line in file:
                z[i] = np.float64(line.rstrip().split("|")[1])
                i += 1
            z = z.reshape((nx, ny))

    fig, ax = subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x, y, z, shade=False, linewidth=1.0, cmap=colormaps["plasma"], antialiased=True)
    show()
