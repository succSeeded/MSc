"""
A blackbox optimizer that uses GP to approximate the function and finds the maximum of its mean. 
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import List
from matplotlib import colormaps
from os import curdir
from os.path import abspath, isfile, join
from time import sleep
from typing import Iterable
from requests import post
from requests.exceptions import ReadTimeout
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt



def plot_colorful_wireframe(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    
    # Normalize to [0,1]
    norm = plt.Normalize(Z.min(), Z.max())
    colors = colormaps["jet"](norm(Z))
    rcount, ccount, _ = colors.shape
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                        facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))



def const_kernel(x: np.ndarray, y: np.ndarray, params: Iterable=[1]) -> np.ndarray:
    """
    Constant kernel.
    """
    c = params[0]
    return c * np.full((max(x.shape), max(y.shape)), c, dtype=np.float64)


def rbf_kernel(x: np.ndarray, y: np.ndarray, params: Iterable=[1,1]) -> np.ndarray:
    """
    Gaussian or Radial Basis Function (RBF) kernel.
    """
    s2, l2 = params
    return s2 * np.exp(-0.5*cdist(x, y, metric="sqeuclidean")/l2)


kernels = {"RBF": rbf_kernel, "Const": const_kernel}
n_params = {"RBF": 2, "Const": 1}



class GaussianProcessRegressor:
    """
    The name is self-explanatory
    """
    def __init__(self, kernel: str="RBF") -> None:
        self._kernel = kernels[kernel]
        self._n_params = n_params[kernel] + 1
        self._noise2 = 1e-6
        
        self._kernel_params = [5, 0.07]


    def neg_log_likelihood(self, theta: Iterable, x: np.ndarray, y: np.ndarray) -> float:
        noise = theta[0]
        kernel_params = theta[1:]
        K = self._kernel(x, x, kernel_params) + noise**2 * np.eye(len(x))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return np.inf

        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
        return (
            0.5 * y.T @ alpha
            + np.sum(np.log(np.diag(L)))
            + 0.5 * len(x) * np.log(2 * np.pi)
        )


    def _optimize_parameters(self, x: np.ndarray, y: np.ndarray) -> None:

        initial_params = [1000.0] * self._n_params
        initial_params[0] = 1e-5
        result = minimize(
            self.neg_log_likelihood,
            initial_params,
            args=(x, y),
            bounds=(((1e-6, None), ) * self._n_params),
            method="L-BFGS-B",
        )
        print(result.x)
        self._noise2 = result.x[0]
        self._kernel_params = result.x[1:]


    def fit(self, x: np.ndarray, y: np.ndarray, optmize_parameters:bool=True) -> None:
        """
        Fit method to precalculate some of the transitional values.
        """
        self._x_train = x
        self._y_train = y
        if optmize_parameters:
            self._optimize_parameters(x, y)
        self._K = self._kernel(x, x, self._kernel_params)
        self._L = np.linalg.cholesky(self._K + self._noise2 * np.eye(self._K.shape[0]))
        self._alpha = solve_triangular(self._L.T, solve_triangular(self._L, y, lower=True))


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
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-f", action="store_true", help="Get the value of true function at optimum"
    )
    varss = vars(parser.parse_args())

    if not isfile(abspath(join(curdir,"fvals.txt"))):
        nx = 11
        ny = 11
        x, y = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
        z = get_grid(x, y)
    else:
        with open("fvals.txt", "r", encoding="utf8") as file:
            content = file.readlines()
            nx, ny = map(np.int64, content[0].rstrip().split(" "))
            x = np.zeros(nx*ny)
            y = np.zeros(nx*ny)
            z = np.zeros(nx*ny)
            for i in range(len(content[1:])):
                coords, fval = content[1 + i].rstrip().split("|")
                x[i], y[i] = map(np.float64, coords.split(" "))
                z[i] = np.float64(fval)

    gpr = GaussianProcessRegressor()
    gpr.fit(np.hstack((x.reshape(-1,1), y.reshape(-1,1))), z, optmize_parameters=False)

    x = x.reshape(nx,ny)
    y = y.reshape(nx,ny)
    z = z.reshape(nx,ny)

    nx_plot = 51j
    ny_plot = 51j
    X, Y = np.mgrid[-1:1:nx_plot, -1:1:ny_plot]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z, _ = gpr.predict(positions.T)
    Z = Z.reshape(int(nx_plot.imag), int(ny_plot.imag))

    f = lambda X: -gpr.predict(X.reshape(1,-1))[0]
    optimum = minimize(f,np.array([0,0]),bounds=((-1,1),(-1,1)),method="L-BFGS-B")
    print(f"Found maximum at: {optimum.x}")

    if varss["f"]:
        post_args = {"secret_key": "WVRjjVBm", "type": "small"}
        with open("optima.txt", "a", encoding="utf8") as file:
            coords = str(optimum.x[0]) + " " + str(optimum.x[1])
            print(f"Requesting value at {coords}\n")

            post_args["x"] = coords

            try:
                response = post("http://optimize-me.ddns.net:8080/", data=post_args, timeout=2)
            except ReadTimeout:
                response = post("http://optimize-me.ddns.net:8080/", data=post_args, timeout=2)

            fval = str(response.content, encoding="utf8")
            file.write(coords + "|" + fval + "\n")
            print(f"True function value: {fval}")

    fig, axs = plt.subplots(ncols=2)
    axs[0].contourf(X, Y, Z, cmap="jet")
    axs[0].set_title("Interpolated fast function")
    axs[0].plot(optimum.x[0], optimum.x[1], "b*")
    axs[1].contourf(x, y, z, cmap="jet")
    axs[1].set_title("Real fast function")
    axs[1].plot(optimum.x[0], optimum.x[1], "b*")

    # plot_colorful_wireframe(X, Y, Z)

    plt.show()
