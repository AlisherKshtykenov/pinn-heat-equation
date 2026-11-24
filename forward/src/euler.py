"""Явная схема Эйлера (FTCS) для уравнения теплопроводности, написана вручную."""
import numpy as np

from .problem import BETA, T, U_LEFT, UX_RIGHT, initial_condition


def solve_euler(n_intervals=40, n_steps=600, neumann_order=1, store_every=1, beta=BETA, t_end=T):
    """Решает задачу и возвращает x, times, U (строка U соответствует моменту времени).

    n_intervals: число отрезков по x, узлов на один больше.
    n_steps: число шагов по времени.
    neumann_order: 1 это U_J = U_{J-1} + 2 dx, 2 это фиктивный узел за границей.
    store_every: сохранять каждый такой шаг, чтобы не держать в памяти все слои.
    """
    dx = 1.0 / n_intervals
    dt = t_end / n_steps
    s = beta * dt / dx**2  # число Куранта для диффузии, для устойчивости нужно s <= 1/2

    x = np.linspace(0.0, 1.0, n_intervals + 1)
    u = initial_condition(x)  # слой m = 0 берём точно из начального условия
    saved_u, saved_t = [u.copy()], [0.0]

    for m in range(1, n_steps + 1):
        new = u.copy()
        # Внутренние узлы: U_j^{m+1} = U_j^m + s (U_{j-1}^m - 2 U_j^m + U_{j+1}^m)
        new[1:-1] = u[1:-1] + s * (u[:-2] - 2.0 * u[1:-1] + u[2:])
        new[0] = U_LEFT  # левая граница: условие Дирихле
        if neumann_order == 1:
            # Односторонняя разность: (U_J - U_{J-1}) / dx = 2, точность O(dx)
            new[-1] = new[-2] + UX_RIGHT * dx
        elif neumann_order == 2:
            # Фиктивный узел U_{J+1} = U_{J-1} + 2 dx * 2, точность O(dx^2)
            new[-1] = u[-1] + 2.0 * s * (u[-2] - u[-1] + UX_RIGHT * dx)
        else:
            raise ValueError("neumann_order должен быть 1 или 2")
        u = new
        if m % store_every == 0:
            saved_u.append(u.copy())
            saved_t.append(m * dt)

    return x, np.array(saved_t), np.array(saved_u)


def diffusion_number(n_intervals, n_steps, beta=BETA, t_end=T):
    """Считает s = beta dt / dx^2 для заданной сетки."""
    return beta * (t_end / n_steps) * n_intervals**2


def profile_at(times, field, t):
    """Достаёт профиль на ближайшем сохранённом слое и проверяет, что время совпало."""
    index = int(np.argmin(np.abs(times - t)))
    if abs(times[index] - t) > 1e-9:
        raise ValueError(f"момент t={t} не сохранён")
    return field[index]
