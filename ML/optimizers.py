from functools import partial
import matplotlib.pyplot as plt
import numpy as np


def golden_section(f, low: float, high: float, tol: float = 1e-6, maxiter: int = 1000):
    """
    Golden section optimizer made strictly for 1D optimization.
    """

    t_inv = 1 / ((1 + np.sqrt(5)) * 0.5)  # inverse golden ratio
    x_1 = low + (1 - t_inv) * (high - low)
    x_2 = low + t_inv * (high - low)
    l = high - low
    n_iter = 2
    f_1 = f(x_1)  # inital function values
    f_2 = f(x_2)

    while True:
        if f_1 > f_2:
            low = x_1
            x_1 = x_2
            x_2 = low + high - x_1
            f_1 = f_2
            f_2 = f(x_2)
        else:
            high = x_2
            x_2 = x_1
            x_1 = low + high - x_2
            f_2 = f_1
            f_1 = f(x_1)

        n_iter += 1

        l = high - low

        # breakpoint criteria
        if l < tol:
            termination = "achieved the step length lower than tolerance"
            break

        if n_iter == maxiter:
            termination = "achieved the maximal number of iterations"
            break

    return {"x": (low + high) * 0.5, "termination": termination}


def powell_optimizer(
    optfunc,
    x_0: np.array,
    max_delta: float = 1.2,
    tol: float = 1e-6,
    maxiter: int = 1000,
) -> dict:
    """
    A simple implementation of [Powell's conjugate direction method](https://en.wikipedia.org/wiki/Powell%27s_method).
    """
    n_iter = 0
    if x_0.shape != (x_0.shape[0],):
        raise ValueError(
            f"Incorrect x_0 shape. It has to be {(x_0.shape,)}, got {x_0.shape} instead"
        )

    x_k = x_0.copy()
    x_record = x_0.copy().reshape(-1, 1)
    basis = np.eye(x_0.shape[0], dtype=np.float64)

    def phi(step, func=None, x0: np.ndarray = None, u_i: np.ndarray = None):
        """
        Local 1D optimization function for Powell method.
        """
        return func(x0 + step * u_i)

    while True:
        n_iter += 1

        x_prev = x_k.copy()

        # nullify the changes to the basis we are stepping in every n steps
        # so that the algorithm is less likely to converge on a local minimum
        if n_iter % 5 == 0:
            basis = np.eye(x_0.shape[0], dtype=np.float64)

        # first, we make steps to the least possible value in the direction of each vector in the basis
        for i in range(x_k.shape[0]):
            step = golden_section(
                partial(phi, func=optfunc, x0=x_k, u_i=basis[:, i]),
                low=-max_delta,
                high=max_delta,
            )["x"]
            x_k += step * basis[:, i]

        # then we save the total of all those steps and make it the last vector in the new basis
        # the vector that is being swapped can be changed to the vector with the most contribution to the p_k
        p_k = x_k - x_prev
        basis[:, :-1] = basis[:, 1:]
        basis[:, -1] = p_k.flatten()

        step = golden_section(
            partial(phi, func=optfunc, x0=x_k, u_i=p_k), low=-max_delta, high=max_delta
        )["x"]
        x_k += step * p_k

        x_record = np.append(x_record, x_k.reshape(-1, 1))

        # breakpoint criteria
        if np.linalg.norm(x_k - x_prev) < tol:
            termination = "achieved the step length lower than tolerance"
            break

        if n_iter == maxiter:
            termination = "achieved the maximal number of iterations"
            break

    return {
        "x": x_k,
        "x_record": x_record.reshape(x_record.shape[0] // x_k.shape[0], x_k.shape[0]),
        "termination": termination,
    }


if __name__ == "__main__":
    # A test on Rosenbrock function

    def rosenbrock(x):
        """
        Rosenbrock function with `alpha` = 200
        """
        if x.shape == (x.shape[0],):
            ans = 200.0 * (x[0] ** 2 - x[1]) ** 2 + (1 - x[1]) ** 2
        else:
            ans = 200.0 * (x[0, :] ** 2 - x[1, :]) ** 2 + (1 - x[0, :]) ** 2
        return ans

    X, Y = np.meshgrid(np.linspace(-5, 5, 1000), np.linspace(-5, 5, 1000))
    print(rosenbrock(np.array([X, Y])).shape)
    plt.contour(X, Y, np.log(rosenbrock(np.array([X, Y]))))
    res = powell_optimizer(rosenbrock, np.array([-1.0, -2.0]))
    print(res)
    plt.scatter(res["x_record"][:, 0], res["x_record"][:, 1], c="black", s=2.0)
    plt.plot(res["x_record"][:, 0], res["x_record"][:, 1], c="black", linewidth=1.0)
    plt.scatter(-1.0, -2.0, c="r")
    plt.show()
