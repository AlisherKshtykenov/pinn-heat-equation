"""Метрики сравнения с эталоном на общей сетке."""
import numpy as np


def max_abs_error(pred, ref):
    """Максимальная абсолютная ошибка: max_i |pred_i - ref_i|."""
    return float(np.max(np.abs(np.asarray(pred) - np.asarray(ref))))


def rmse(pred, ref):
    """Среднеквадратичная ошибка: sqrt(1/M * sum_i (pred_i - ref_i)^2)."""
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(ref)) ** 2)))
