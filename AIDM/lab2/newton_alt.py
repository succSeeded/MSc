def golden_section(f, low: float, high: float, tol: float = 1e-6, maxiter: int = 1_000):
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

@njit
def approximate_jacobian(func, omegas, dx):
    """
    Approximate the Jacobi matrix
    """
    return np.array(
        [
            (func(omegas + np.array([dx, 0.0])) - func(omegas)) / dx,
            (func(omegas + np.array([0.0, dx])) - func(omegas)) / dx,
        ]
    )


def approximate_grad(func, omegas, dx):
    """
    Approximate the gradient
    """
    return func(omegas) * approximate_jacobian(func, omegas, dx)


# @njit
def approximate_hessian(func, omegas, dx):
    """
    Approximate hessian calculation
    """
    dxx = (
        (
            func(omegas + np.array([dx, 0.0]))
            - 2.0 * func(omegas)
            + func(omegas - np.array([dx, 0.0]))
        )
        / dx
        / dx
    )
    dyy = (
        (
            func(omegas + np.array([0.0, dx]))
            - 2.0 * func(omegas)
            + func(omegas - np.array([0.0, dx]))
        )
        / dx
        / dx
    )
    dxy = (
        (
            func(omegas + np.array([dx, dx]))
            - func(omegas + np.array([dx, 0.0]))
            - func(omegas + np.array([0.0, dx]))
            + func(omegas)
        )
        / dx
        / dx
    )

    return np.array([[dxx, dxy], [dxy, dyy]])
    # return approximate_jacobian(func, omegas, dx).reshape(-1, 1) @ approximate_jacobian(
    #     func, omegas, dx
    # ).reshape(1, -1)


def newton_solver(
    target_fn,
    omegas_init,
    dx=1e-3,
    tol=1e-6,
    max_iter=1_000,
    verbose=True,
):
    """
    Implement Newton method to find optimal frequencies omega1, omega2
    """

    omega12_curr = np.array(omegas_init)
    iter_id = 0

    while iter_id <= max_iter:
        omega12_prev = omega12_curr

        step_direction = -(
            np.linalg.inv(
                approximate_hessian(target_fn, omega12_prev, dx) + 1e-9 * np.eye(2)
            )
            @ approximate_grad(target_fn, omega12_prev, dx)
        )
        learning_rate = golden_section(
            lambda l: target_fn(omega12_prev + l * step_direction), 0.0, 1.0
        )["x"]

        omega12_curr = omega12_prev + step_direction * learning_rate

        if verbose:
            print(f"Approximate step: {step_direction * learning_rate}")
            print(f"Current parameters: {omega12_curr}")
            print(f"Current target function value: {target_fn(omega12_prev)}")
            print("\n" + "=" * 80 + "\n\n")

        iter_id += 1
        if target_fn(omega12_curr) < tol:
            if verbose:
                print(f"Functional is close to the minimum. Terminating...")
            break

    return omega12_curr