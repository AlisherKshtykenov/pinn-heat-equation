"""Запуски PINN с кэшем: обученный запуск лежит в weights/ и повторно не считается."""
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path

import torch

from .pinn import PINN, DTYPE, PINNConfig, train_pinn

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"
SEEDS = (0, 1, 2)
BOUNDARY_WEIGHTS = (0.1, 1.0, 10.0, 100.0)  # значения веса границ в эксперименте D3


def all_runs():
    """Полный список запусков работы: имя файла и настройки."""
    runs = {f"main_seed{seed}": PINNConfig(seed=seed) for seed in SEEDS}
    # Рекомендованный минимум: те же настройки, но 1000 итераций
    runs["baseline_1000"] = PINNConfig(iterations=1000)
    # Без масштабирования физики: вес невязки 1 вместо T^2
    runs["no_physics_weight"] = PINNConfig(weight_physics=1.0)
    # Контроль: новые случайные точки на каждой итерации
    runs["resample_seed0"] = PINNConfig(resample=True)
    # D3: меняем только вес двух граничных потерь
    for weight in BOUNDARY_WEIGHTS:
        if weight == 1.0:
            continue  # этот вариант совпадает с main_seed*
        for seed in SEEDS:
            runs[f"boundary{weight:g}_seed{seed}"] = PINNConfig(
                seed=seed, weight_dirichlet=weight, weight_neumann=weight
            )
    return runs


def run_path(name):
    """Путь к файлу запуска."""
    return WEIGHTS_DIR / f"{name}.pt"


def train_and_save(name, cfg):
    """Обучает один запуск на одном потоке и сохраняет веса, историю и время."""
    torch.set_num_threads(1)  # один поток, чтобы время было сравнимо с кодом Эйлера на numpy
    model, history, seconds = train_pinn(cfg)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": asdict(cfg), "history": history, "seconds": seconds},
        run_path(name),
    )
    return name, seconds


def _worker(item):
    return train_and_save(*item)


def ensure_trained(processes=4):
    """Обучает только те запуски, которых ещё нет. Возвращает список обученных сейчас."""
    runs = all_runs()
    done = []
    # main_seed0 считается один, без параллельной нагрузки: его время идёт в отчёт
    if not run_path("main_seed0").exists():
        done.append(train_and_save("main_seed0", runs["main_seed0"]))
    missing = [(name, cfg) for name, cfg in runs.items() if not run_path(name).exists()]
    if missing:
        # spawn нужен, потому что на macOS fork вместе с torch ненадёжен
        with mp.get_context("spawn").Pool(processes) as pool:
            for result in pool.imap_unordered(_worker, missing):
                done.append(result)
                print("обучено:", *result, flush=True)
    return done


def load_run(name):
    """Загружает запуск и возвращает модель и словарь с настройками, историей и временем."""
    run = torch.load(run_path(name), weights_only=False)
    config = PINNConfig(**run["config"])
    model = PINN(config.hidden_layers, config.hidden_units).to(DTYPE)
    model.load_state_dict(run["state_dict"])
    model.eval()
    return model, run
