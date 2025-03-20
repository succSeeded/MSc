"""
A blackbox optimizer that uses GP to approximate the function and finds the maximum of its mean.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import List
from matplotlib.axes import Axes
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numpy.random import uniform
from os import curdir
from os.path import abspath, isfile, join
from time import sleep
from typing import Iterable
from requests import post
from requests.exceptions import ReadTimeout
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


def pretty_wireframe(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, ax: Axes = None
) -> None:
    """
    A version of wireframe that supports multicoloured output. Essentially a surface with `alpha` = 0.
    """
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    # Normalize to [0,1]
    norm = plt.Normalize(Z.min(), Z.max())
    colors = colormaps["jet"](norm(Z))
    rcount, ccount, _ = colors.shape
    surf = ax.plot_surface(
        X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False
    )
    surf.set_facecolor((0, 0, 0, 0))


def const_kernel(x: np.ndarray, y: np.ndarray, params: Iterable = [1]) -> np.ndarray:
    """
    Constant kernel.
    """
    c = params[0]
    return c * np.full((max(x.shape), max(y.shape)), c, dtype=np.float64)


def rbf_kernel(x: np.ndarray, y: np.ndarray, params: Iterable = [1, 1]) -> np.ndarray:
    """
    Gaussian or Radial Basis Function (RBF) kernel.
    """
    s2, l2 = params
    return s2 * np.exp(-0.5 * cdist(x, y, metric="sqeuclidean") / l2)


kernels = {"RBF": rbf_kernel, "Const": const_kernel}
n_params = {"RBF": 2, "Const": 1}


class GaussianProcessRegressor:
    """
    The name is self-explanatory
    """

    def __init__(self, kernel: str = "RBF") -> None:
        self._kernel = kernels[kernel]
        self._n_params = n_params[kernel]
        self._noise2 = 1e-10
        self._kernel_params = np.array([5, 0.07])

    def neg_log_likelihood(
        self, theta: Iterable, x: np.ndarray, y: np.ndarray
    ) -> float:

        # noise2 = theta[-1]
        # kernel_params = theta[:-1]

        noise2 = self._noise2
        kernel_params = theta
        K = self._kernel(x, x, kernel_params) + noise2 * np.eye(len(x))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return np.inf

        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
        return (
            0.5 * y.T @ alpha
            + 0.5 * np.log(np.sum(np.diag(L)))
            + 0.5 * len(x) * np.log(2 * np.pi)
        )

    def _optimize_parameters(self, x: np.ndarray, y: np.ndarray) -> None:

        # initial_params = np.concantenate((self._kernel_params, [self._noise2]))
        initial_params = self._kernel_params
        result = minimize(
            self.neg_log_likelihood,
            initial_params,
            args=(x, y),
            bounds=(((1e-10, 1000.0),) * self._n_params),
            method="Powell",
        )
        # self._noise2 = result.x[-1]
        # self._kernel_params = result.x[:-1]

        self._kernel_params = result.x
        print(result.x)

    def fit(
        self, x: np.ndarray, y: np.ndarray, optmize_parameters: bool = True
    ) -> None:
        """
        Fit method to precalculate some of the transitional values.
        """
        self._x_train = x
        self._y_train = y
        if optmize_parameters:
            self._optimize_parameters(x, y)
        self._K = self._kernel(x, x, self._kernel_params)
        self._L = np.linalg.cholesky(self._K + self._noise2 * np.eye(self._K.shape[0]))
        self._alpha = solve_triangular(
            self._L.T, solve_triangular(self._L, y, lower=True)
        )

    def predict(self, x: np.ndarray) -> List:
        """
        Make a prediction based on training data
        """
        k_star = self._kernel(self._x_train, x, self._kernel_params)
        k_starstar = self._kernel(x, x, self._kernel_params)
        mu = k_star.T @ self._alpha
        v = solve_triangular(self._L, k_star, lower=True)
        cov = k_starstar - v.T @ v
        return mu, cov


def get_fval(x: float, y: float, func_type:str="small") -> float:
    """
    Get function value at a specific point using POST request.
    """
    post_args = {"secret_key": "WVRjjVBm", "type": func_type}
    with open(f"fvals_{func_type}.txt", "a", encoding="utf8") as file:
        coords = f"{x} {y}"
        print(f"Requesting value at {coords}\n")

        post_args["x"] = coords

        try:
            response = post(
                "http://optimize-me.ddns.net:8080/", data=post_args, timeout=2.0
            )
        except ReadTimeout:
            response = post(
                "http://optimize-me.ddns.net:8080/", data=post_args, timeout=2.0
            )

        fval = str(response.content, encoding="utf8")
        z = np.float64(response.content)
        file.write(coords + "|" + fval + "\n")

        sleep(1)
    return z


def get_grid(x_coords: np.ndarray, y_coords: np.ndarray, func_type:str="small") -> np.ndarray:
    """
    Get function values on the grid using POST requests.
    """
    post_args = {"secret_key": "WVRjjVBm", "type": func_type}
    z = np.zeros_like(x_coords)

    with open(f"fvals_{func_type}.txt", "a", encoding="utf8") as file:
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                coords = f"{x_coords[i, j]} {y_coords[i, j]}"
                print(f"Requesting value at {coords}\n")

                post_args["x"] = coords

                try:
                    response = post(
                        "http://optimize-me.ddns.net:8080/", data=post_args, timeout=2.0
                    )
                except ReadTimeout:
                    response = post(
                        "http://optimize-me.ddns.net:8080/", data=post_args, timeout=2.0
                    )

                fval = str(response.content, encoding="utf8")
                z[i, j] = np.float64(response.content)
                file.write(coords + "|" + fval + "\n")

                sleep(1)
    return z


def get_points_from_file(fname: str) -> list:
    """
    Get a set of points and F(x,y) for those points from a file.
    """
    with open(fname, "r", encoding="utf8") as file:
        content = file.readlines()
        npoints = len(content[:])
        x = np.zeros(npoints)
        y = np.zeros(npoints)
        z = np.zeros(npoints)
        for i in range(len(content[:])):
            coords, fval = content[i].rstrip().split("|")
            x[i], y[i] = map(np.float64, coords.split(" "))
            z[i] = np.float64(fval)
    return [x, y, z]


def get_randpoints(npoints: int, func_type:str="small") -> list:
    """
    Get function values at `npoints` random points using POST requests.
    """
    post_args = {"secret_key": "WVRjjVBm", "type": func_type}
    x, y = uniform(-1, 1, size=(2, npoints))
    z = np.zeros(npoints)

    with open(f"fvals_{func_type}.txt", "a", encoding="utf8") as file:
        for i in range(npoints):
            coords = f"{x[i]} {y[i]}"
            print(f"Requesting value at {coords}")

            post_args["x"] = coords

            try:
                response = post(
                    "http://optimize-me.ddns.net:8080/", data=post_args, timeout=2.0
                )
                print("OK!")
            except ReadTimeout:
                response = post(
                    "http://optimize-me.ddns.net:8080/", data=post_args, timeout=2.0
                )
                print("OK!")

            fval = str(response.content, encoding="utf8")
            z[i] = np.float64(response.content)
            file.write(coords + "|" + fval + "\n")

            sleep(2)
    return [x, y, z]


def EI_acquisition(fval: float, mu: float, std: float) -> float:
    """
    Expected Improvement(EI) acquisition function.
    Parameters:
        `fval`(float): function value at a certain point,
        `mu`(float): mean of a surrogate model at a certain point,
        `fval`(float): standard deviation of a surrogate model at a certain point,
    """
    
    standard_normal_cdf = norm.cdf
    standard_normal_pdf = norm.pdf
    return (mu - fval) * standard_normal_cdf((mu - fval) / std) + std * standard_normal_pdf((mu - fval) / std)


if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-f", action="store_true", help="Get the value of true function at optimum"
    )
    parser.add_argument(
        "-a", action="store_true", help="Add new points"
    )
    parser.add_argument(
        "--plot_surf", action="store_true", help="Plot wireframe of the interpolant"
    )
    parser.add_argument(
        "--plot_contours",
        action="store_true",
        help="Plot contours of the fast function and its interpolant",
    )
    varss = vars(parser.parse_args())

    if not isfile(abspath(join(curdir, "fvals_large.txt"))) or varss["a"]:
        x, y, z = get_randpoints(10)
    else:
        x, y, z = get_points_from_file("fvals_large.txt")

    points = np.hstack((x.flatten().reshape(-1, 1), y.flatten().reshape(-1, 1)))

    gpr = GaussianProcessRegressor()
    gpr.fit(
        points,
        z.flatten().reshape(-1, 1),
        optmize_parameters=True,
    )

    curr_optimum = np.max(z)
    curr_optimum_point = points[np.argmax(z),:]
    print(curr_optimum_point)

    def target(point: np.ndarray, curr_optimum: float):
        mu, std = gpr.predict(point.reshape(1, -1))
        return EI_acquisition(curr_optimum, mu, std)

    n_bayes_iters = 10
    for i in range(n_bayes_iters):
        rpoint = uniform(-1, 1, size=2)
        optimum = minimize(target, rpoint, args=(curr_optimum), bounds=((-1, 1), (-1, 1)), method="L-BFGS-B")
        curr_optimum_point = optimum.x
        ei_optimum = optimum.fun
        print(curr_optimum_point)
        print(ei_optimum)

    # def target(X: np.ndarray) -> float:
    #     return -gpr.predict(X.reshape(1, -1))[0]

    # rpoint = np.array([-0.20, -0.25])
    # optimum = minimize(target, rpoint, bounds=((-1, 1), (-1, 1)), method="L-BFGS-B")
    # for i in range(4):
    #     rpoint = uniform(-1, 1, size=2)
    #     curr_opt = minimize(
    #         target, rpoint, bounds=((-1, 1), (-1, 1)), method="L-BFGS-B"
    #     )
    #     optimum = curr_opt if target(curr_opt.x) < target(optimum.x) else optimum

    # print(f"Found maximum at: {optimum.x}")
    # print(f"Mean F = {-target(optimum.x)}")
    # print(f"std F = {gpr.predict(optimum.x.reshape(1,-1))[1]}")

    if varss["f"]:
        post_args = {"secret_key": "WVRjjVBm", "type": "large"}
        with open("fvals_true.txt", "a", encoding="utf8") as file:
            coords = str(optimum.x[0]) + " " + str(optimum.x[1])
            print(f"Requesting value at {coords}\n")

            post_args["x"] = coords

            try:
                response = post(
                    "http://optimize-me.ddns.net:8080/", data=post_args, timeout=2
                )
            except ReadTimeout:
                response = post(
                    "http://optimize-me.ddns.net:8080/", data=post_args, timeout=2
                )

            fval = str(response.content, encoding="utf8")
            file.write(coords + "|" + fval + "\n")
            print(f"True function value: {float(response.content):0.6f}")

    if varss["plot_contours"] or varss["plot_surf"]:
        nx_plot = 101j
        ny_plot = 101j
        X, Y = np.mgrid[-1:1:nx_plot, -1:1:ny_plot]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z, _ = gpr.predict(positions.T)
        Z = Z.reshape(int(nx_plot.imag), int(ny_plot.imag))

        if varss["plot_contours"]:

            fig, axs = plt.subplots(ncols=1)
            contours = axs.contourf(X, Y, Z, cmap="jet")
            axs.set_title("Interpolated fast function")
            # axs.plot(optimum.x[0], optimum.x[1], "b*")
            fig.colorbar(ScalarMappable(norm=Normalize(np.min(Z), np.max(Z)), cmap="jet"), ax=axs, orientation="vertical")

        if varss["plot_surf"]:
            pretty_wireframe(X, Y, Z)

        plt.show()
