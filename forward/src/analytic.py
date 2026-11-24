"""Аналитическое решение: u(x, t) = u_s(x) + сумма ряда по собственным функциям."""
import numpy as np

from .problem import BETA, steady_state


def wave_numbers(n_terms):
    """Волновые числа k_n = (n + 1/2) * pi для n = 0, ..., n_terms - 1."""
    return (np.arange(n_terms) + 0.5) * np.pi


def series_coefficients(n_terms):
    """Коэффициенты b_n = (-1)^(n+1) * 4 pi / (4 pi^2 - k_n^2)."""
    k = wave_numbers(n_terms)
    signs = (-1.0) ** (np.arange(n_terms) + 1)
    return signs * 4.0 * np.pi / (4.0 * np.pi**2 - k**2)


def transient(x, t, n_terms=2000, beta=BETA):
    """Переходная часть v(x, t) как усечённый ряд из n_terms слагаемых."""
    x = np.asarray(x, dtype=float)
    k = wave_numbers(n_terms)
    b = series_coefficients(n_terms)
    # Строки матрицы соответствуют точкам x, столбцы соответствуют номерам n
    return np.sin(np.outer(x, k)) @ (b * np.exp(-beta * k**2 * t))


def exact_solution(x, t, n_terms=2000, beta=BETA):
    """Полное решение u(x, t) = u_s(x) + v(x, t) в момент времени t."""
    x = np.asarray(x, dtype=float)
    return steady_state(x) + transient(x, t, n_terms, beta)
