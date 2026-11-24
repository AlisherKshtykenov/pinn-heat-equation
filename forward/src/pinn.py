"""PINN для уравнения теплопроводности. Точное решение в обучении не используется."""
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .problem import BETA, T, U_LEFT, UX_RIGHT

DTYPE = torch.float64  # двойная точность: физическая невязка очень мала по модулю


@dataclass
class PINNConfig:
    """Все настройки одного запуска, чтобы их можно было записать в отчёт."""

    hidden_layers: int = 3
    hidden_units: int = 32
    n_interior: int = 2000  # точек коллокации внутри области
    n_initial: int = 200  # точек при tau = 0
    n_boundary: int = 200  # точек на каждой из границ x = 0 и x = 1
    iterations: int = 20000
    lr: float = 1e-3  # начальный шаг Adam
    lr_final: float = 1e-5  # конечный шаг при косинусном снижении
    weight_physics: float = T**2  # такой вес равен обезразмеренной невязке u_tau - beta T u_xx
    weight_initial: float = 1.0
    weight_dirichlet: float = 1.0
    weight_neumann: float = 1.0
    resample: bool = False  # True: новые случайные точки на каждой итерации
    seed: int = 0
    log_every: int = 10


class PINN(nn.Module):
    """Полносвязная сеть (x, tau) -> u с активацией tanh."""

    def __init__(self, hidden_layers=3, hidden_units=32):
        super().__init__()
        layers = [nn.Linear(2, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.Tanh()]
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)  # начальный масштаб весов для tanh
                nn.init.zeros_(layer.bias)

    def forward(self, x, tau):
        # Переводим входы из [0, 1] в [-1, 1], tanh так обучается заметно легче
        inputs = torch.cat([2.0 * x - 1.0, 2.0 * tau - 1.0], dim=1)
        return self.net(inputs)


def initial_tensor(x):
    """Начальное условие для тензоров, копия формулы из problem.py на torch."""
    return 2.0 * x + torch.sin(2.0 * torch.pi * x) + 1.0


def gradients(u, x, tau=None, second=False):
    """Автоматическое дифференцирование: возвращает u_x, а также u_tau и u_xx по запросу."""
    # create_graph=True строит граф для производных, чтобы по ним шёл backward к весам
    if tau is not None:
        u_x, u_tau = torch.autograd.grad(u.sum(), (x, tau), create_graph=True)
    else:
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_tau = None
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0] if second else None
    return u_x, u_tau, u_xx


def physics_residual(model, x, tau, beta=BETA, t_end=T):
    """Невязка в физическом масштабе: r = (1 / T) u_tau - beta u_xx."""
    x = x.clone().requires_grad_(True)
    tau = tau.clone().requires_grad_(True)
    u = model(x, tau)
    _, u_tau, u_xx = gradients(u, x, tau, second=True)
    return u_tau / t_end - beta * u_xx


def sample_points(cfg, generator):
    """Случайные равномерные точки для всех четырёх частей функции потерь."""

    def uniform(n):
        return torch.rand(n, 1, generator=generator, dtype=DTYPE)

    return {
        "x_r": uniform(cfg.n_interior),
        "tau_r": uniform(cfg.n_interior),
        "x_ic": uniform(cfg.n_initial),
        "tau_b": uniform(cfg.n_boundary),
    }


def loss_terms(model, points):
    """Считает четыре среднеквадратичных слагаемых функции потерь."""
    # 1. Уравнение внутри области
    residual = physics_residual(model, points["x_r"], points["tau_r"])
    loss_physics = (residual**2).mean()

    # 2. Начальное условие при tau = 0
    x_ic = points["x_ic"]
    pred_ic = model(x_ic, torch.zeros_like(x_ic))
    loss_initial = ((pred_ic - initial_tensor(x_ic)) ** 2).mean()

    # 3. Левая граница: u(0, tau) = 1
    tau_b = points["tau_b"]
    pred_left = model(torch.zeros_like(tau_b), tau_b)
    loss_dirichlet = ((pred_left - U_LEFT) ** 2).mean()

    # 4. Правая граница: u_x(1, tau) = 2
    x_right = torch.ones_like(tau_b).requires_grad_(True)
    pred_right = model(x_right, tau_b)
    u_x, _, _ = gradients(pred_right, x_right)
    loss_neumann = ((u_x - UX_RIGHT) ** 2).mean()

    return {
        "physics": loss_physics,
        "initial": loss_initial,
        "dirichlet": loss_dirichlet,
        "neumann": loss_neumann,
    }


def train_pinn(cfg):
    """Обучает сеть и возвращает модель, историю потерь и время обучения в секундах."""
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    model = PINN(cfg.hidden_layers, cfg.hidden_units).to(DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.iterations, eta_min=cfg.lr_final)

    weights = {
        "physics": cfg.weight_physics,
        "initial": cfg.weight_initial,
        "dirichlet": cfg.weight_dirichlet,
        "neumann": cfg.weight_neumann,
    }
    points = sample_points(cfg, generator)
    history = {"iteration": [], "total": [], "physics": [], "initial": [], "dirichlet": [], "neumann": []}

    start = time.perf_counter()
    for iteration in range(1, cfg.iterations + 1):
        if cfg.resample:
            points = sample_points(cfg, generator)
        optimizer.zero_grad()
        terms = loss_terms(model, points)
        total = sum(weights[name] * value for name, value in terms.items())
        total.backward()  # backward идёт сквозь все производные к весам сети
        optimizer.step()
        scheduler.step()

        if iteration == 1 or iteration % cfg.log_every == 0:
            history["iteration"].append(iteration)
            history["total"].append(total.item())
            for name, value in terms.items():
                history[name].append(value.item())
    train_seconds = time.perf_counter() - start

    return model, {key: np.array(value) for key, value in history.items()}, train_seconds


def predict(model, x, t, t_end=T):
    """Предсказание на сетке x (numpy) в физический момент времени t."""
    x_tensor = torch.as_tensor(np.asarray(x), dtype=DTYPE).reshape(-1, 1)
    tau_tensor = torch.full_like(x_tensor, t / t_end)
    with torch.no_grad():
        return model(x_tensor, tau_tensor).numpy().ravel()


def boundary_residuals(model, times, t_end=T):
    """Возвращает u(0, t) - 1 и u_x(1, t) - 2 для массива моментов времени."""
    tau = torch.as_tensor(np.asarray(times) / t_end, dtype=DTYPE).reshape(-1, 1)
    left = model(torch.zeros_like(tau), tau).detach().numpy().ravel() - U_LEFT
    x_right = torch.ones_like(tau).requires_grad_(True)
    u_x, _, _ = gradients(model(x_right, tau), x_right)
    return left, u_x.detach().numpy().ravel() - UX_RIGHT
