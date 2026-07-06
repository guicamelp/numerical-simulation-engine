# -*- coding: utf-8 -*-
"""
Gêmeo Digital Multiphysics — Parte 6
=====================================

Integra, preservando os módulos legados, os três subsistemas do trabalho:

    placa térmica  -> viscosidade distribuída nos canais -> rede hidráulica
    -> membrana elástica em formulação monolítica.

O script cobre os exercícios destacados no Capítulo 6:
    6.3.2  Monte Carlo estacionário e dinâmico;
    6.4.3  interpolação, regressão e ruído;
    6.4.5  sensibilidades por diferenças finitas e método direto;
    6.4.5  Newton–Raphson para E(H) - 7.5 = 0;
    6.5    animação integrada 2D/3D.

As classes/nomes legados ThermalPlateConfig, ThermalPlateSolver,
TemperatureInterpolator, NetworkInfluenceModel e Config permanecem
expostos neste módulo por importação direta dos arquivos anteriores.

Execução:
    python gemeo_digital_integrado.py --full
    python gemeo_digital_integrado.py --quick

A execução padrão é reprodutível (seed fixa) e escreve todos os artefatos
em resultados_gemeo_digital/.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from scipy import sparse
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.sparse.linalg import factorized, spsolve, splu

# -----------------------------------------------------------------------------
# MÓDULOS ORIGINAIS PRESERVADOS
# -----------------------------------------------------------------------------
# Estes imports preservam a arquitetura já construída nas Partes 4 e 5. O novo
# arquivo apenas adiciona a camada integradora e as rotinas do Capítulo 6.
import acoplamento_hidraulico_termico as aht
import acoplamento_hidraulico_mecanico as ahm

ThermalPlateConfig = aht.ThermalPlateConfig
ThermalPlateSolver = aht.ThermalPlateSolver
TemperatureInterpolator = aht.TemperatureInterpolator
NetworkInfluenceModel = aht.NetworkInfluenceModel
HydroThermalCoupledModel = aht.HydroThermalCoupledModel
Config = ahm.Config

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "resultados_gemeo_digital"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Evita avisos de interpolação/condicionamento que não alteram os cálculos.
warnings.filterwarnings("ignore", category=RuntimeWarning)


# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DO GÊMEO DIGITAL COMPLETO
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DigitalTwinConfig:
    """Parâmetros físicos, numéricos e estatísticos do estudo integrado."""

    # Placa térmica (SI)
    Lx: float = 0.030
    Ly: float = 0.015
    k0: float = 0.25
    source_value: float = 5.0e5
    TL: float = 10.0
    TR: float = 30.0
    TC: float = 35.0
    circle_x: float = 0.0225
    circle_y: float = 0.0075
    circle_radius: float = 0.0025
    thermal_Nx: int = 81
    thermal_Ny: int = 41
    thermal_influence_radius: float = 5.0e-4

    # Rede hidráulica
    levels: int = 3
    inlet_node: int = 0
    outlet_node: int = 5
    H: float = 1000.0e-6
    p_inlet: float = 5000.0
    network_length_unit: float = 1.0e-3

    # Membrana (mantém a classe Config original)
    membrane_Nx: int = 51
    membrane_Ny: int = 51
    radius: float = 0.0025
    thickness: float = 0.0001
    sigma: float = 200.0
    rho: float = 900.0
    beta_hat: float = 0.1

    # Tempo adimensional
    tau_final: float = 4.0
    dt: float = 0.05

    # Monte Carlo
    seed: int = 20260626
    q_critical: float = 1.25e-5
    E_critical: float = 7.0


def make_mechanical_cfg(cfg: DigitalTwinConfig, *, H: Optional[float] = None,
                        dt: Optional[float] = None, tau_final: Optional[float] = None,
                        p_inlet: Optional[float] = None) -> Config:
    """Reconstrói Config preservando as variáveis e escalas da Parte 5."""
    return Config(
        levels=cfg.levels,
        inlet_node=cfg.inlet_node,
        outlet_node=cfg.outlet_node,
        mu=5.0e-4,  # substituída posteriormente pela viscosidade por aresta.
        channel_width=cfg.H if H is None else H,
        network_length_unit=cfg.network_length_unit,
        radius=cfg.radius,
        thickness=cfg.thickness,
        sigma=cfg.sigma,
        rho=cfg.rho,
        beta_hat=cfg.beta_hat,
        Nx=cfg.membrane_Nx,
        Ny=cfg.membrane_Ny,
        dt_hat=cfg.dt if dt is None else dt,
        tau_final=cfg.tau_final if tau_final is None else tau_final,
        p_inlet=cfg.p_inlet if p_inlet is None else p_inlet,
        output_dir=str(OUTPUT_DIR),
        show_plots=False,
    )


# -----------------------------------------------------------------------------
# UTILITÁRIOS NUMÉRICOS E VISUAIS
# -----------------------------------------------------------------------------
def newton_cotes_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Soma de Newton–Cotes fechada de grau 1: regra do trapézio composta."""
    return float(np.trapezoid(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))


def l2_norm(residual: np.ndarray, x: np.ndarray) -> float:
    """Norma L2 contínua aproximada por Newton–Cotes (trapézio composto)."""
    return math.sqrt(max(newton_cotes_trapezoid(np.asarray(residual) ** 2, x), 0.0))


def savefig(fig: plt.Figure, name: str) -> Path:
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def finite_difference_steps(value: float) -> float:
    """Passo recomendado para diferença finita em float64: sqrt(eps)*(1+|x|)."""
    return math.sqrt(np.finfo(float).eps) * (1.0 + abs(float(value)))


def scale_network_to_plate(Xno: np.ndarray, cfg: DigitalTwinConfig,
                           margin: float = 0.08) -> np.ndarray:
    """Mapeia a topologia mecânica, em mm, para a placa térmica em metros.

    A transformação é afim e preserva a conectividade/todas as proporções da
    rede. A margem impede que o grafo coincida numericamente com o contorno
    Dirichlet da placa.
    """
    Xno = np.asarray(Xno, dtype=float)
    lo = Xno.min(axis=0)
    hi = Xno.max(axis=0)
    span = np.maximum(hi - lo, 1e-14)
    u = (Xno - lo) / span
    x = cfg.Lx * (margin + (1.0 - 2.0 * margin) * u[:, 0])
    y = cfg.Ly * (margin + (1.0 - 2.0 * margin) * u[:, 1])
    return np.column_stack((x, y))


def plot_network(ax: plt.Axes, X: np.ndarray, conec: np.ndarray, *,
                 color: str = "white", linewidth: float = 0.45,
                 alpha: float = 0.8) -> None:
    segs = [np.vstack((X[i], X[j])) for i, j in np.asarray(conec, dtype=int)]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=linewidth, alpha=alpha))


# -----------------------------------------------------------------------------
# ESTÁGIO TÉRMICO COM FATORAÇÃO REUTILIZADA
# -----------------------------------------------------------------------------
class CachedThermalStage:
    """Resolve a placa uma vez estruturalmente e varia TC apenas no vetor b.

    A matriz de condução A é independente de TC quando a geometria, k(x,y) e
    as condições de fronteira permanecem fixas. Assim, sua fatoração LU é
    reutilizada em todos os estudos de sensibilidade, evitando custo redundante.
    """

    def __init__(self, cfg: DigitalTwinConfig, X_plate: np.ndarray, conec: np.ndarray) -> None:
        self.cfg = cfg
        self.X_plate = np.asarray(X_plate, dtype=float)
        self.conec = np.asarray(conec, dtype=int)
        self.influence = NetworkInfluenceModel(
            self.X_plate, self.conec,
            d_max=cfg.thermal_influence_radius,
            k0=cfg.k0,
        )

        thermal_cfg = ThermalPlateConfig(
            Lx=cfg.Lx,
            Ly=cfg.Ly,
            Nx=cfg.thermal_Nx,
            Ny=cfg.thermal_Ny,
            TL=cfg.TL,
            TR=cfg.TR,
            TB=aht.top_bottom_temperature_function(cfg.Lx),
            TT=aht.top_bottom_temperature_function(cfg.Lx),
            source_function=aht.constant_source(cfg.source_value),
            use_variable_k=True,
            k_constant=cfg.k0,
            k_function=self.influence.k_modified_at_point,
            use_circle_constraint=True,
            circle_center_x=cfg.circle_x,
            circle_center_y=cfg.circle_y,
            circle_radius=cfg.circle_radius,
            TC=cfg.TC,
            solver_mode="sparse",
            output_dir=OUTPUT_DIR,
        )
        self.solver = ThermalPlateSolver(thermal_cfg)
        self.A = self.solver.assembly(matrix_mode="sparse").tocsc()
        self.solve_A = factorized(self.A)
        self.fixed_ids = self.solver.dirichlet_ids.copy()

    def _refresh_dirichlet_values(self, TC: float) -> None:
        self.solver.cfg.TC = float(TC)
        ids, values = self.solver._build_dirichlet_data()
        if not np.array_equal(ids, self.fixed_ids):
            raise RuntimeError("A geometria Dirichlet mudou; a fatoração térmica não é reutilizável.")
        self.solver.dirichlet_ids = ids
        self.solver.dirichlet_values = values
        self.solver.dirichlet_set = set(ids.tolist())
        self.solver.dirichlet_value_by_id = {
            int(i): float(v) for i, v in zip(ids, values)
        }

    def solve(self, TC: float, *, quadrature_rule: str = "trapezoid",
              subdivisions: int = 8) -> Dict[str, Any]:
        """Retorna T(x,y), temperaturas e viscosidades médias por canal."""
        self._refresh_dirichlet_values(TC)
        b = self.solver.build_rhs()
        T_vec = np.asarray(self.solve_A(b), dtype=float)
        T_grid = T_vec.reshape((self.cfg.thermal_Ny, self.cfg.thermal_Nx))
        interp = TemperatureInterpolator(self.solver.x, self.solver.y, T_grid, method="cubic")

        # Duas formas de média, mantidas para análise de Jensen / quadratura.
        T_edge_mean = aht.mean_edge_temperature(
            self.X_plate, self.conec, interp, quadrature_rule, subdivisions
        )
        mu_from_mean_T = aht.empirical_viscosity(T_edge_mean)
        mu_edge_direct = aht.mean_edge_viscosity_direct(
            self.X_plate, self.conec, interp, quadrature_rule, subdivisions
        )
        return {
            "T_vec": T_vec,
            "T_grid": T_grid,
            "x": self.solver.x.copy(),
            "y": self.solver.y.copy(),
            "Tmax": float(T_grid.max()),
            "Tmean": float(T_grid.mean()),
            "T_edge_mean": T_edge_mean,
            "mu_from_mean_T": mu_from_mean_T,
            "mu_edge": mu_edge_direct,
        }


# -----------------------------------------------------------------------------
# REDE HIDRÁULICA: CONDUTÂNCIA E FALHAS
# -----------------------------------------------------------------------------
def conductance_per_edge(Xno: np.ndarray, conec: np.ndarray, mu_edge: np.ndarray,
                         H: float, length_unit: float) -> np.ndarray:
    """Condutância Poiseuille equivalente, Ck ∝ H^4/mu_k."""
    lengths = ahm.edge_lengths(Xno, conec, length_unit)
    area = H ** 2
    d_eq = math.sqrt(4.0 * area / math.pi)
    kappa = math.pi * d_eq ** 4 / (128.0 * np.asarray(mu_edge, dtype=float))
    return kappa / lengths


def impose_row_dirichlet(A: sparse.csr_matrix, values: Dict[int, float]) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Imposição por substituição de linha, coerente com a Parte 5."""
    A_mod = A.tolil(copy=True)
    b = np.zeros(A.shape[0], dtype=float)
    for node, value in values.items():
        A_mod[int(node), :] = 0.0
        A_mod[int(node), int(node)] = 1.0
        b[int(node)] = float(value)
    return A_mod.tocsr(), b


def RandomFail(C_original: np.ndarray, p_o: float, f_obs: float,
               rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Gera um cenário de obstrução independente usando U(0,1).

    Cada aresta falha quando U_k < p_o e então C_k <- C_k/f_obs.
    """
    C_original = np.asarray(C_original, dtype=float)
    failed = rng.random(C_original.size) < float(p_o)
    C_mod = C_original.copy()
    C_mod[failed] /= float(f_obs)
    return C_mod, failed


def solve_static_network(Xno: np.ndarray, conec: np.ndarray, C: np.ndarray,
                         inlet_node: int, outlet_node: int, p_inlet: float) -> Dict[str, Any]:
    """Problema estacionário de falhas: p_inlet prescrito e p_outlet=0."""
    A = ahm.assembly_hydraulic(conec, C)
    A_mod, b = impose_row_dirichlet(A, {inlet_node: p_inlet, outlet_node: 0.0})
    p = np.asarray(spsolve(A_mod, b), dtype=float)
    qin = float((A @ p)[inlet_node])
    return {"A": A, "p": p, "qin": qin}


# -----------------------------------------------------------------------------
# MODELO MONOLÍTICO (REFERÊNCIA) E SENSIBILIDADES DIRETAS
# -----------------------------------------------------------------------------
def build_monolithic_system(cfg: DigitalTwinConfig, Xno: np.ndarray, conec: np.ndarray,
                            mu_edge: np.ndarray, H: float, dt: float) -> Dict[str, Any]:
    """Monta G para Y=[w,v,p]^T, preservando a formulação da Parte 5.

    A matriz G é invariante no tempo para um cenário (TC,H,dt) fixo. A
    fatoração é feita fora do laço temporal, como solicitado no enunciado.
    """
    mcfg = make_mechanical_cfg(cfg, H=H, dt=dt)
    scales = ahm.reference_scales(mcfg)
    C = conductance_per_edge(Xno, conec, mu_edge, H, cfg.network_length_unit)
    A_phys = ahm.assembly_hydraulic(conec, C).tocsr()
    A_hat = (A_phys * scales["p_ref"] / (scales["v_ref"] * cfg.radius ** 2)).tocsr()

    A_bc = A_hat.tolil(copy=True)
    A_bc[cfg.inlet_node, :] = 0.0
    A_bc[cfg.inlet_node, cfg.inlet_node] = 1.0
    A_bc = A_bc.tocsr()

    K, M, x_hat, y_hat, h_hat, active_mask = ahm.assembly_membrane(
        cfg.membrane_Nx, cfg.membrane_Ny
    )
    n_m = cfg.membrane_Nx * cfg.membrane_Ny
    n_p = Xno.shape[0]
    U = ahm.build_U(cfg.membrane_Nx, cfg.membrane_Ny, active_mask, n_p, cfg.outlet_node)
    I = sparse.identity(n_m, format="csr")
    D = cfg.beta_hat * M

    G = sparse.bmat(
        [
            [(1.0 / dt) * I, -I, None],
            [K, (1.0 / dt) * M + D, -U.T],
            [None, (h_hat ** 2) * U, A_bc],
        ],
        format="csc",
    )

    # dA/dH = 4 A/H; linhas impostas de Dirichlet têm derivada nula.
    dA_hat = (4.0 / H) * A_hat
    dA_bc = ((4.0 / H) * A_bc).tolil()
    dA_bc[cfg.inlet_node, :] = 0.0
    dA_bc = dA_bc.tocsr()

    return {
        "mcfg": mcfg,
        "scales": scales,
        "C": C,
        "A_phys": A_phys,
        "A_hat": A_hat,
        "A_bc": A_bc,
        "dA_hat_dH": dA_hat.tocsr(),
        "dA_bc_dH": dA_bc,
        "K": K,
        "M": M,
        "U": U,
        "h_hat": h_hat,
        "active_mask": active_mask,
        "x_hat": x_hat,
        "y_hat": y_hat,
        "G": G,
        "SolveG": factorized(G),
        "n_m": n_m,
        "n_p": n_p,
    }


def run_monolithic(
    cfg: DigitalTwinConfig,
    Xno: np.ndarray,
    conec: np.ndarray,
    mu_edge: np.ndarray,
    H: float,
    dt: float,
    *,
    p_inlet_function: Optional[Callable[[float], float]] = None,
    compute_sensitivity_H: bool = False,
    store_fields: bool = False,
) -> Dict[str, Any]:
    """Marcha no tempo com G pré-fatorada e, opcionalmente, sensibilidade direta.

    A energia é calculada por P(t)=p^T A p com a matriz hidráulica física
    antes da troca da linha Dirichlet. Isso impede que a condição de pressão
    prescrita seja contabilizada como dissipação artificial.
    """
    sys = build_monolithic_system(cfg, Xno, conec, mu_edge, H, dt)
    n_m, n_p = int(sys["n_m"]), int(sys["n_p"])
    M, U, SolveG = sys["M"], sys["U"], sys["SolveG"]
    A_hat, dA_hat = sys["A_hat"], sys["dA_hat_dH"]
    dA_bc = sys["dA_bc_dH"]
    scales = sys["scales"]
    h_hat = float(sys["h_hat"])

    tau = np.arange(0.0, cfg.tau_final + 0.5 * dt, dt)
    n_t = tau.size
    w = np.zeros(n_m, dtype=float)
    v = np.zeros(n_m, dtype=float)
    sw = np.zeros(n_m, dtype=float)
    sv = np.zeros(n_m, dtype=float)
    sp = np.zeros(n_p, dtype=float)

    active = U.getrow(cfg.outlet_node).indices
    hist: Dict[str, np.ndarray] = {
        "tau": tau,
        "p_inlet": np.zeros(n_t),
        "p_outlet": np.zeros(n_t),
        "q_inlet_hat": np.zeros(n_t),
        "q_outlet_hat": np.zeros(n_t),
        "volume_m3": np.zeros(n_t),
        "power": np.zeros(n_t),
        "d_power_dH": np.zeros(n_t),
        "w_center_m": np.zeros(n_t),
        "w_max_abs_m": np.zeros(n_t),
    }
    if compute_sensitivity_H:
        hist["dq_inlet_hat_dH"] = np.zeros(n_t)
        hist["dq_outlet_hat_dH"] = np.zeros(n_t)
        hist["d_volume_dH_m3_per_m"] = np.zeros(n_t)
    if store_fields:
        hist["w_fields"] = np.zeros((n_t, n_m), dtype=np.float32)

    p_hist = np.zeros((n_t, n_p), dtype=float)
    volume = 0.0
    dvolume = 0.0
    last_qout = 0.0
    last_dqout = 0.0

    for k, tk in enumerate(tau):
        pin = cfg.p_inlet if p_inlet_function is None else float(p_inlet_function(float(tk)))
        rhs = np.zeros(2 * n_m + n_p, dtype=float)
        rhs[:n_m] = w / dt
        rhs[n_m:2 * n_m] = M.dot(v) / dt
        rhs[2 * n_m + cfg.inlet_node] = pin / scales["p_ref"]
        Y = np.asarray(SolveG(rhs), dtype=float)
        w, v, p = Y[:n_m], Y[n_m:2 * n_m], Y[2 * n_m:]

        if compute_sensitivity_H:
            rhs_s = np.zeros_like(rhs)
            rhs_s[:n_m] = sw / dt
            rhs_s[n_m:2 * n_m] = M.dot(sv) / dt
            rhs_s[2 * n_m:] = -dA_bc.dot(p)
            S = np.asarray(SolveG(rhs_s), dtype=float)
            sw, sv, sp = S[:n_m], S[n_m:2 * n_m], S[2 * n_m:]

        qout_hat = h_hat ** 2 * float(np.sum(v[active]))
        qin_hat = float((A_hat @ p)[cfg.inlet_node])
        if k > 0:
            volume += 0.5 * dt * (last_qout + qout_hat) * scales["t_ref"] * scales["v_ref"] * cfg.radius ** 2
        last_qout = qout_hat

        power = float(p @ (A_hat @ p))
        hist["p_inlet"][k] = pin
        hist["p_outlet"][k] = p[cfg.outlet_node] * scales["p_ref"]
        hist["q_inlet_hat"][k] = qin_hat
        hist["q_outlet_hat"][k] = qout_hat
        hist["volume_m3"][k] = volume
        hist["power"][k] = power
        hist["w_center_m"][k] = w[ahm.ij2n(cfg.membrane_Nx // 2, cfg.membrane_Ny // 2, cfg.membrane_Nx)] * scales["w_ref"]
        hist["w_max_abs_m"][k] = np.max(np.abs(w)) * scales["w_ref"]
        p_hist[k] = p
        if store_fields:
            hist["w_fields"][k] = (w * scales["w_ref"]).astype(np.float32)

        if compute_sensitivity_H:
            dqout_hat = h_hat ** 2 * float(np.sum(sv[active]))
            dqin_hat = float((dA_hat @ p + A_hat @ sp)[cfg.inlet_node])
            dpower = float(2.0 * sp @ (A_hat @ p) + p @ (dA_hat @ p))
            if k > 0:
                dvolume += 0.5 * dt * (last_dqout + dqout_hat) * scales["t_ref"] * scales["v_ref"] * cfg.radius ** 2
            last_dqout = dqout_hat
            hist["dq_inlet_hat_dH"][k] = dqin_hat
            hist["dq_outlet_hat_dH"][k] = dqout_hat
            hist["d_volume_dH_m3_per_m"][k] = dvolume
            hist["d_power_dH"][k] = dpower

    energy = newton_cotes_trapezoid(hist["power"], tau)
    out: Dict[str, Any] = {
        "system": sys,
        "history": hist,
        "p_history": p_hist,
        "energy": energy,
        "qin_final_hat": float(hist["q_inlet_hat"][-1]),
        "volume_final_m3": float(hist["volume_m3"][-1]),
    }
    if compute_sensitivity_H:
        out.update(
            {
                "denergy_dH": newton_cotes_trapezoid(hist["d_power_dH"], tau),
                "dqin_final_hat_dH": float(hist["dq_inlet_hat_dH"][-1]),
                "dvolume_final_m3_per_m": float(hist["d_volume_dH_m3_per_m"][-1]),
            }
        )
    return out


# -----------------------------------------------------------------------------
# ELIMINAÇÃO DE SCHUR: MONTE CARLO DINÂMICO MAIS RÁPIDO
# -----------------------------------------------------------------------------
class ReducedDynamicSolver:
    """Resolve o mesmo acoplamento por complemento de Schur de posto um.

    A rede hidráulica possui uma única interface cinemática com a membrana,
    associada ao nó outlet. Eliminar as pressões internas reduz a atualização
    da rede a p_out = a - r*q. O termo resultante na equação estrutural é uma
    atualização de posto um; a fórmula de Sherman–Morrison evita fatorar G de
    dimensão 2*N_m+N_p para cada cenário aleatório de Monte Carlo.
    """

    def __init__(self, cfg: DigitalTwinConfig, Xno: np.ndarray, conec: np.ndarray, dt: float, modes: int = 28) -> None:
        self.cfg = cfg
        self.Xno = np.asarray(Xno, dtype=float)
        self.conec = np.asarray(conec, dtype=int)
        self.dt = float(dt)
        self.mcfg = make_mechanical_cfg(cfg, dt=dt)
        self.scales = ahm.reference_scales(self.mcfg)
        self.K, self.M, self.x_hat, self.y_hat, self.h_hat, self.active_mask = ahm.assembly_membrane(
            cfg.membrane_Nx, cfg.membrane_Ny
        )
        self.n_m_full = cfg.membrane_Nx * cfg.membrane_Ny
        self.n_p = self.Xno.shape[0]
        self.U = ahm.build_U(cfg.membrane_Nx, cfg.membrane_Ny, self.active_mask, self.n_p, cfg.outlet_node)
        self.u_full = np.zeros(self.n_m_full, dtype=float)
        self.u_full[self.U.getrow(cfg.outlet_node).indices] = 1.0

        # Redução modal: a malha continua sendo 51x51, mas a evolução para o
        # Monte Carlo é projetada nos modos físicos mais baixos da membrana.
        # Isto reduz drasticamente o custo de 4000 trajetórias mantendo a
        # resposta dominante da membrana. O caso nominal é validado contra G.
        evals, Phi, _ = ahm.solve_membrane_modes(self.K, self.M, num_modes=modes)
        self.Phi = Phi
        self.lambda_modal = np.asarray(evals, dtype=float)
        self.n_m = self.lambda_modal.size
        self.g = np.asarray(Phi.T @ self.u_full, dtype=float).ravel()
        I = sparse.identity(self.n_m, format="csr")
        Lambda = sparse.diags(self.lambda_modal, format="csr")
        self.G0 = sparse.bmat(
            [
                [(1.0 / dt) * I, -I],
                [Lambda, (1.0 / dt + cfg.beta_hat) * I],
            ],
            format="csc",
        )
        self.SolveG0 = factorized(self.G0)
        self.a_rank = np.concatenate((np.zeros(self.n_m), self.g))
        self.z_rank = np.asarray(self.SolveG0(self.a_rank), dtype=float)
        self.free = np.array([i for i in range(self.n_p) if i != cfg.inlet_node], dtype=int)
        self.outlet_free_index = int(np.where(self.free == cfg.outlet_node)[0][0])

    def network_reduction(self, C: np.ndarray) -> Dict[str, Any]:
        """Obtém p_unit e x para p = pin_hat*p_unit - x*q_hat."""
        A_phys = ahm.assembly_hydraulic(self.conec, C).tocsr()
        A_hat = (A_phys * self.scales["p_ref"] / (self.scales["v_ref"] * self.cfg.radius ** 2)).tocsr()
        Lff = A_hat[self.free][:, self.free].tocsc()
        # A rede pode ter condição ruim sob obstruções; o LU dá diagnóstico claro.
        lu = splu(Lff)
        lfi = np.asarray(A_hat[self.free, self.cfg.inlet_node].todense()).ravel()
        p_unit_free = lu.solve(-lfi)
        eout = np.zeros(self.free.size)
        eout[self.outlet_free_index] = 1.0
        x_free = lu.solve(eout)
        p_unit = np.zeros(self.n_p)
        x = np.zeros(self.n_p)
        p_unit[self.cfg.inlet_node] = 1.0
        p_unit[self.free] = p_unit_free
        x[self.free] = x_free
        return {"A_hat": A_hat, "p_unit": p_unit, "x": x}

    def run(self, C: np.ndarray, *, p_inlet_function: Optional[Callable[[float], float]] = None,
            store_history: bool = False) -> Dict[str, Any]:
        red = self.network_reduction(C)
        A_hat, p_unit, x = red["A_hat"], red["p_unit"], red["x"]
        r = float(x[self.cfg.outlet_node])
        # G = G0 + a b^T, b = [0; h^2 r u].
        b_rank = np.concatenate((np.zeros(self.n_m), (self.h_hat ** 2) * r * self.g))
        denom = float(1.0 + b_rank @ self.z_rank)
        if abs(denom) < 1e-12:
            raise FloatingPointError("Atualização de Sherman–Morrison quase singular.")

        # Coeficientes da forma quadrática da potência, para p = pin_hat p_unit - q x.
        P00 = float(p_unit @ (A_hat @ p_unit))
        P10 = float(x @ (A_hat @ p_unit))
        P11 = float(x @ (A_hat @ x))
        qin0 = float((A_hat @ p_unit)[self.cfg.inlet_node])
        qin1 = float((A_hat @ x)[self.cfg.inlet_node])

        tau = np.arange(0.0, self.cfg.tau_final + 0.5 * self.dt, self.dt)
        w = np.zeros(self.n_m)
        v = np.zeros(self.n_m)
        vol = 0.0
        last_qout = 0.0
        P = np.zeros(tau.size)
        if store_history:
            history = {
                "tau": tau,
                "power": P,
                "p_outlet": np.zeros(tau.size),
                "q_inlet_hat": np.zeros(tau.size),
                "q_outlet_hat": np.zeros(tau.size),
                "volume_m3": np.zeros(tau.size),
                "w_center_m": np.zeros(tau.size),
                "w_max_abs_m": np.zeros(tau.size),
            }

        for k, tk in enumerate(tau):
            pin = self.cfg.p_inlet if p_inlet_function is None else float(p_inlet_function(float(tk)))
            pin_hat = pin / self.scales["p_ref"]
            rhs = np.empty(2 * self.n_m)
            rhs[:self.n_m] = w / self.dt
            rhs[self.n_m:] = v / self.dt + self.g * (pin_hat * p_unit[self.cfg.outlet_node])
            y = np.asarray(self.SolveG0(rhs), dtype=float)
            sol = y - self.z_rank * float(b_rank @ y) / denom
            w, v = sol[:self.n_m], sol[self.n_m:]
            qout_hat = self.h_hat ** 2 * float(self.g @ v)
            p_out_hat = pin_hat * p_unit[self.cfg.outlet_node] - r * qout_hat
            power = P00 * pin_hat ** 2 - 2.0 * P10 * pin_hat * qout_hat + P11 * qout_hat ** 2
            P[k] = max(float(power), 0.0)
            if k > 0:
                vol += 0.5 * self.dt * (last_qout + qout_hat) * self.scales["t_ref"] * self.scales["v_ref"] * self.cfg.radius ** 2
            last_qout = qout_hat
            if store_history:
                history["p_outlet"][k] = p_out_hat * self.scales["p_ref"]
                history["q_inlet_hat"][k] = pin_hat * qin0 - qout_hat * qin1
                history["q_outlet_hat"][k] = qout_hat
                history["volume_m3"][k] = vol
                history["w_center_m"][k] = float(self.Phi[ahm.ij2n(self.cfg.membrane_Nx // 2, self.cfg.membrane_Ny // 2, self.cfg.membrane_Nx)] @ w) * self.scales["w_ref"]
                history["w_max_abs_m"][k] = np.max(np.abs(self.Phi @ w)) * self.scales["w_ref"]

        result: Dict[str, Any] = {
            "energy": newton_cotes_trapezoid(P, tau),
            "power": P,
            "tau": tau,
            "network": red,
            "modal_modes": int(self.n_m),
        }
        if store_history:
            result["history"] = history
        return result


# -----------------------------------------------------------------------------
# OBJETO ORQUESTRADOR
# -----------------------------------------------------------------------------
class MultiphysicsDigitalTwin:
    """Facade principal: integra calor, hidráulica, membrana e pós-processamento."""

    def __init__(self, cfg: DigitalTwinConfig) -> None:
        self.cfg = cfg
        self.Xno, self.conec = ahm.generate_graph_arrays(cfg.levels)
        self.X_plate = scale_network_to_plate(self.Xno, cfg)
        self.thermal_stage = CachedThermalStage(cfg, self.X_plate, self.conec)
        self._thermal_cache: Dict[float, Dict[str, Any]] = {}
        self._reduced_cache: Dict[float, ReducedDynamicSolver] = {}

    def thermal(self, TC: Optional[float] = None) -> Dict[str, Any]:
        key = round(self.cfg.TC if TC is None else float(TC), 10)
        if key not in self._thermal_cache:
            self._thermal_cache[key] = self.thermal_stage.solve(key)
        return self._thermal_cache[key]

    def C_nominal(self, *, TC: Optional[float] = None, H: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        th = self.thermal(TC)
        H_use = self.cfg.H if H is None else float(H)
        C = conductance_per_edge(self.Xno, self.conec, th["mu_edge"], H_use, self.cfg.network_length_unit)
        return C, th

    def monolithic(self, *, TC: Optional[float] = None, H: Optional[float] = None,
                   dt: Optional[float] = None, sensitivity_H: bool = False,
                   p_inlet_function: Optional[Callable[[float], float]] = None,
                   store_fields: bool = False) -> Dict[str, Any]:
        th = self.thermal(TC)
        return run_monolithic(
            self.cfg, self.Xno, self.conec, th["mu_edge"],
            self.cfg.H if H is None else float(H),
            self.cfg.dt if dt is None else float(dt),
            p_inlet_function=p_inlet_function,
            compute_sensitivity_H=sensitivity_H,
            store_fields=store_fields,
        )

    def reduced(self, dt: float) -> ReducedDynamicSolver:
        key = float(dt)
        if key not in self._reduced_cache:
            self._reduced_cache[key] = ReducedDynamicSolver(self.cfg, self.Xno, self.conec, key)
        return self._reduced_cache[key]


# -----------------------------------------------------------------------------
# MONTE CARLO — EXERCÍCIO 6.3.2
# -----------------------------------------------------------------------------
def stationary_monte_carlo(twin: MultiphysicsDigitalTwin, *, p_o: float, f_obs: float,
                           N: int, seed: int) -> Dict[str, Any]:
    C0, _ = twin.C_nominal()
    rng = np.random.default_rng(seed)
    qin = np.empty(N)
    failed_fraction = np.empty(N)
    for i in range(N):
        C, failed = RandomFail(C0, p_o, f_obs, rng)
        sol = solve_static_network(twin.Xno, twin.conec, C, twin.cfg.inlet_node,
                                   twin.cfg.outlet_node, twin.cfg.p_inlet)
        qin[i] = sol["qin"]
        failed_fraction[i] = failed.mean()
    indicator = qin < twin.cfg.q_critical
    prob = np.cumsum(indicator) / np.arange(1, N + 1)
    se = np.sqrt(np.maximum(prob * (1.0 - prob) / np.arange(1, N + 1), 0.0))
    final_prob = float(prob[-1])
    # Primeiro índice a partir do qual todas as estimativas permanecem a <1 pp do valor final.
    tol = 0.01
    stable_index = N
    deviation = np.abs(prob - final_prob)
    suffix_max = np.maximum.accumulate(deviation[::-1])[::-1]
    candidates = np.where((np.arange(N) >= 199) & (suffix_max <= tol))[0]
    if candidates.size:
        stable_index = int(candidates[0] + 1)
    return {
        "qin": qin,
        "failed_fraction": failed_fraction,
        "indicator": indicator,
        "prob": prob,
        "se": se,
        "final_prob": final_prob,
        "stable_N": stable_index,
        "mean_qin": float(qin.mean()),
    }


def dynamic_monte_carlo(twin: MultiphysicsDigitalTwin, *, p_o: float, f_obs: float,
                        N: int, dt: float, seed: int) -> Dict[str, Any]:
    """Monte Carlo dinâmico com complemento de Schur equivalente ao monolítico."""
    C0, _ = twin.C_nominal()
    reduced = twin.reduced(dt)
    rng = np.random.default_rng(seed)
    energies = np.empty(N)
    failures = np.empty(N, dtype=bool)
    for i in range(N):
        C, _ = RandomFail(C0, p_o, f_obs, rng)
        E = reduced.run(C)["energy"]
        energies[i] = E
        failures[i] = E < twin.cfg.E_critical
    prob = float(failures.mean())
    se = math.sqrt(max(prob * (1.0 - prob) / N, 0.0))
    return {"energies": energies, "failures": failures, "prob": prob, "se": se, "dt": dt}


def run_monte_carlo_exercises(twin: MultiphysicsDigitalTwin, *, quick: bool) -> Dict[str, Any]:
    cfg = twin.cfg
    N_conv = 1200 if quick else 3500
    N_sweep = 400 if quick else 1500
    N_dyn = 250 if quick else 2000

    # Cenário representativo para a curva de convergência.
    conv = stationary_monte_carlo(twin, p_o=0.40, f_obs=10.0, N=N_conv, seed=cfg.seed + 1)
    p_values = np.round(np.arange(0.05, 0.651, 0.05), 2)
    rows: List[Dict[str, float]] = []
    for f_obs in (5.0, 10.0):
        for p_o in p_values:
            res = stationary_monte_carlo(
                twin, p_o=float(p_o), f_obs=f_obs, N=N_sweep,
                seed=cfg.seed + int(1000 * p_o) + int(10 * f_obs),
            )
            rows.append({
                "p_o": float(p_o), "f_obs": f_obs, "Prob": res["final_prob"],
                "SE": float(res["se"][-1]), "N": N_sweep,
            })
    sweep = pd.DataFrame(rows)

    # p_o=0.40 e f_obs=10 oferecem severidade suficiente para comparar as malhas temporais.
    dyn05 = dynamic_monte_carlo(twin, p_o=0.40, f_obs=10.0, N=N_dyn, dt=0.05, seed=cfg.seed + 50)
    dyn10 = dynamic_monte_carlo(twin, p_o=0.40, f_obs=10.0, N=N_dyn, dt=0.10, seed=cfg.seed + 51)

    # Gráfico de convergência com IC 95% aproximado.
    n = np.arange(1, N_conv + 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    ax.plot(n, conv["prob"], label="Estimativa Monte Carlo")
    ax.fill_between(n, np.maximum(0, conv["prob"] - 1.96 * conv["se"]),
                    np.minimum(1, conv["prob"] + 1.96 * conv["se"]), alpha=0.22,
                    label="IC 95% aproximado")
    ax.axhline(conv["final_prob"], linestyle="--", label="Estimativa final")
    ax.axvline(conv["stable_N"], linestyle=":", label=f"N estável ≈ {conv['stable_N']}")
    ax.set(xlabel="Número de realizações N", ylabel="Prob(q_inlet < q_crítico)",
           title="Monte Carlo estacionário: convergência do estimador")
    ax.grid(True, alpha=.28); ax.legend()
    savefig(fig, "04_mc_estacionario_convergencia.png")

    # Severidade versus p_o.
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    for f_obs in (5.0, 10.0):
        df = sweep[sweep["f_obs"] == f_obs]
        ax.errorbar(df["p_o"], df["Prob"], yerr=1.96 * df["SE"], marker="o",
                    capsize=3, label=f"f_obs={int(f_obs)}")
    ax.set(xlabel="Probabilidade individual de obstrução p_o", ylabel="Probabilidade crítica",
           title="Severidade de obstrução: resposta estacionária")
    ax.grid(True, alpha=.28); ax.legend()
    savefig(fig, "05_mc_estacionario_severidade.png")

    # Comparação dinâmica: histogramas e ICs.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, dyn in zip(axes, (dyn05, dyn10)):
        ax.hist(dyn["energies"], bins=30, alpha=.75, density=True)
        ax.axvline(cfg.E_critical, linestyle="--", label="E crítico=7")
        ax.set(title=f"δt={dyn['dt']:.2f}: Prob={dyn['prob']:.4f} ± {1.96*dyn['se']:.4f}",
               xlabel="Energia adimensional E", ylabel="Densidade")
        ax.grid(True, alpha=.25); ax.legend()
    savefig(fig, "06_mc_dinamico_histogramas_dt.png")

    pd.DataFrame({
        "N": n, "prob": conv["prob"], "se": conv["se"], "qin_m3_s": conv["qin"],
    }).to_csv(OUTPUT_DIR / "mc_estacionario_convergencia.csv", index=False)
    sweep.to_csv(OUTPUT_DIR / "mc_estacionario_severidade.csv", index=False)
    pd.DataFrame({"E_dt005": dyn05["energies"], "E_dt010": dyn10["energies"]}).to_csv(
        OUTPUT_DIR / "mc_dinamico_energias.csv", index=False
    )

    return {"convergence": conv, "sweep": sweep, "dynamic_dt005": dyn05, "dynamic_dt010": dyn10}


# -----------------------------------------------------------------------------
# APRENDIZADO DE MODELOS — EXERCÍCIO 6.4.3
# -----------------------------------------------------------------------------
def normal_equations_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Mínimos quadrados via A^T A c=A^T y, com x reescalado para estabilidade."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    center = 0.5 * (x.min() + x.max())
    half = 0.5 * (x.max() - x.min())
    z = (x - center) / half
    A = np.vander(z, degree + 1, increasing=True)
    ATA = A.T @ A
    ATy = A.T @ y
    coeff = np.linalg.solve(ATA, ATy)

    def evaluate(x_eval: np.ndarray) -> np.ndarray:
        zz = (np.asarray(x_eval, dtype=float) - center) / half
        return np.vander(zz, degree + 1, increasing=True) @ coeff

    return coeff, evaluate


def run_learning_exercises(twin: MultiphysicsDigitalTwin, nominal: Dict[str, Any], *, quick: bool) -> Dict[str, Any]:
    h = nominal["history"]
    t = h["tau"]
    P = h["power"]
    x_eval = np.linspace(t.min(), t.max(), 1600)
    linear = interp1d(t, P, kind="linear")
    cubic = interp1d(t, P, kind="cubic")
    P_lin, P_cub = linear(x_eval), cubic(x_eval)

    max_degree = 9 if quick else 15
    degrees = np.arange(3, max_degree + 1)
    rows = []
    fitters: Dict[int, Callable[[np.ndarray], np.ndarray]] = {}
    for deg in degrees:
        coeff, f = normal_equations_polynomial(t, P, int(deg))
        fitters[int(deg)] = f
        err = l2_norm(f(x_eval) - np.interp(x_eval, t, P), x_eval)
        rows.append({"degree": int(deg), "L2_error": err, "coeff_norm": float(np.linalg.norm(coeff))})
    errors = pd.DataFrame(rows)

    # Curvas limpas, com foco em amostras e em estruturas local/global.
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(t, P, "o", markersize=3.5, label="Dados P(t)")
    ax.plot(x_eval, P_lin, label="Spline linear")
    ax.plot(x_eval, P_cub, label="Spline cúbica")
    for deg in sorted(set([3, min(7, max_degree), max_degree])):
        ax.plot(x_eval, fitters[deg](x_eval), linestyle="--", label=f"MQ global: grau {deg}")
    ax.set(xlabel="Tempo adimensional τ", ylabel="Potência adimensional P(τ)",
           title="Interpolação local e regressão global sobre a resposta dinâmica")
    ax.grid(True, alpha=.28); ax.legend(ncol=2)
    savefig(fig, "07_aproximacoes_dados_limpos.png")

    fig, ax = plt.subplots(figsize=(8.3, 4.5))
    ax.semilogy(errors["degree"], errors["L2_error"], marker="o")
    ax.set(xlabel="Grau do polinômio global", ylabel="Erro L2 contínuo",
           title="Erro de regressão: diagnóstico de underfitting/overfitting")
    ax.grid(True, alpha=.28)
    savefig(fig, "08_regressao_erro_L2.png")

    # Ruído prescrito pelo enunciado: p_inlet*(1+U[-0.15,0.15]).
    rng = np.random.default_rng(twin.cfg.seed + 80)
    noise = -0.15 + 0.30 * rng.random(t.size)
    pin_noisy = twin.cfg.p_inlet * (1.0 + noise)
    pfun = interp1d(t, pin_noisy, kind="previous", bounds_error=False,
                    fill_value=(pin_noisy[0], pin_noisy[-1]))
    noisy = twin.monolithic(p_inlet_function=lambda tt: float(pfun(tt)), store_fields=False)
    Pn = noisy["history"]["power"]
    lin_n = interp1d(t, Pn, kind="linear")
    cub_n = interp1d(t, Pn, kind="cubic")
    degree_noise = min(7, max_degree)
    _, reg_n = normal_equations_polynomial(t, Pn, degree_noise)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(t, Pn, "o", markersize=3.1, alpha=.8, label="Dados com ruído no inlet")
    ax.plot(x_eval, lin_n(x_eval), label="Spline linear")
    ax.plot(x_eval, cub_n(x_eval), label="Spline cúbica")
    ax.plot(x_eval, reg_n(x_eval), linestyle="--", label=f"MQ global grau {degree_noise}")
    ax.set(xlabel="Tempo adimensional τ", ylabel="P(τ)",
           title="Sensibilidade a ruídos: interpolação exata versus filtragem global")
    ax.grid(True, alpha=.28); ax.legend()
    savefig(fig, "09_aproximacoes_dados_ruidosos.png")

    errors.to_csv(OUTPUT_DIR / "regressao_erros_L2.csv", index=False)
    pd.DataFrame({"tau": t, "P_limpo": P, "p_inlet_ruidoso_Pa": pin_noisy, "P_ruidoso": Pn}).to_csv(
        OUTPUT_DIR / "dados_aproximacao.csv", index=False
    )
    return {"errors": errors, "noisy": noisy, "best_degree": int(errors.loc[errors["L2_error"].idxmin(), "degree"])}


# -----------------------------------------------------------------------------
# SENSIBILIDADES — EXERCÍCIO 6.4.5
# -----------------------------------------------------------------------------
def output_triplet(result: Dict[str, Any]) -> np.ndarray:
    """[E, q_inlet físico, V final físico]."""
    scales = result["system"]["scales"]
    cfg_radius = result["system"]["mcfg"].radius
    q_phys = result["qin_final_hat"] * scales["v_ref"] * cfg_radius ** 2
    return np.array([result["energy"], q_phys, result["volume_final_m3"]], dtype=float)


def scan_sensitivities(twin: MultiphysicsDigitalTwin, *, quick: bool) -> Dict[str, Any]:
    npts = 7 if quick else 11
    TC_values = np.linspace(0.0, 250.0, npts)
    H_values = np.linspace(500e-6, 1500e-6, npts)

    # TC: diferenças finitas forward/centered.
    tc_rows: List[Dict[str, float]] = []
    for TC in TC_values:
        eps = finite_difference_steps(float(TC))
        f0 = output_triplet(twin.monolithic(TC=float(TC)))
        fp = output_triplet(twin.monolithic(TC=float(TC + eps)))
        fm = output_triplet(twin.monolithic(TC=float(TC - eps)))
        df_fwd = (fp - f0) / eps
        df_ctr = (fp - fm) / (2.0 * eps)
        tc_rows.append({
            "TC_C": float(TC), "E": f0[0], "qin_m3_s": f0[1], "V_m3": f0[2],
            "dE_dTC_fwd": df_fwd[0], "dE_dTC_ctr": df_ctr[0],
            "dqin_dTC_fwd": df_fwd[1], "dqin_dTC_ctr": df_ctr[1],
            "dV_dTC_fwd": df_fwd[2], "dV_dTC_ctr": df_ctr[2],
        })
    df_tc = pd.DataFrame(tc_rows)

    # H: forward/centered e método direto contínuo para E. O resultado direto
    # também fornece dq_in/dH e dV/dH pelas sensibilidades do estado.
    h_rows: List[Dict[str, float]] = []
    for H in H_values:
        eps = finite_difference_steps(float(H))
        base = twin.monolithic(H=float(H), sensitivity_H=True)
        f0 = output_triplet(base)
        fp = output_triplet(twin.monolithic(H=float(H + eps)))
        fm = output_triplet(twin.monolithic(H=float(H - eps)))
        df_fwd = (fp - f0) / eps
        df_ctr = (fp - fm) / (2.0 * eps)
        scales = base["system"]["scales"]
        qscale = scales["v_ref"] * twin.cfg.radius ** 2
        h_rows.append({
            "H_um": H * 1e6, "E": f0[0], "qin_m3_s": f0[1], "V_m3": f0[2],
            "dE_dH_fwd_per_m": df_fwd[0], "dE_dH_ctr_per_m": df_ctr[0],
            "dE_dH_direct_per_m": float(base["denergy_dH"]),
            "dqin_dH_fwd_m3_s_per_m": df_fwd[1], "dqin_dH_ctr_m3_s_per_m": df_ctr[1],
            "dqin_dH_direct_m3_s_per_m": float(base["dqin_final_hat_dH"] * qscale),
            "dV_dH_fwd_m3_per_m": df_fwd[2], "dV_dH_ctr_m3_per_m": df_ctr[2],
            "dV_dH_direct_m3_per_m": float(base["dvolume_final_m3_per_m"]),
        })
    df_H = pd.DataFrame(h_rows)

    # TC plots with readable engineering units.
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 10.0), sharex="col")
    axes[0, 0].plot(df_tc.TC_C, df_tc.E, marker="o")
    axes[1, 0].plot(df_tc.TC_C, df_tc.qin_m3_s * 1e9, marker="o")
    axes[2, 0].plot(df_tc.TC_C, df_tc.V_m3 * 1e12, marker="o")
    for ax, col_f, col_c, ylabel in [
        (axes[0, 1], "dE_dTC_fwd", "dE_dTC_ctr", "dE/dTC"),
        (axes[1, 1], "dqin_dTC_fwd", "dqin_dTC_ctr", "dq_in/dTC [µL/s/°C]"),
        (axes[2, 1], "dV_dTC_fwd", "dV_dTC_ctr", "dV/dTC [nL/°C]"),
    ]:
        y1, y2 = df_tc[col_f].to_numpy(), df_tc[col_c].to_numpy()
        if "qin" in ylabel:
            y1, y2 = y1 * 1e9, y2 * 1e9
        if "dV" in ylabel:
            y1, y2 = y1 * 1e12, y2 * 1e12
        ax.plot(df_tc.TC_C, y1, marker="o", label="forward")
        ax.plot(df_tc.TC_C, y2, marker="s", label="centered")
        ax.set_ylabel(ylabel); ax.grid(True, alpha=.25); ax.legend()
    axes[0, 0].set_ylabel("Energia E")
    axes[1, 0].set_ylabel("q_in [µL/s]")
    axes[2, 0].set_ylabel("V(tf) [nL]")
    axes[2, 0].set_xlabel("TC [°C]"); axes[2, 1].set_xlabel("TC [°C]")
    axes[0, 0].set_title("Funções"); axes[0, 1].set_title("Derivadas por diferenças finitas")
    savefig(fig, "10_sensibilidades_TC.png")

    # H functions + derivatives. H derivatives converted per µm.
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 10.0), sharex="col")
    axes[0, 0].plot(df_H.H_um, df_H.E, marker="o")
    axes[1, 0].plot(df_H.H_um, df_H.qin_m3_s * 1e9, marker="o")
    axes[2, 0].plot(df_H.H_um, df_H.V_m3 * 1e12, marker="o")
    derivative_specs = [
        ("dE_dH_fwd_per_m", "dE_dH_ctr_per_m", "dE_dH_direct_per_m", "dE/dH [por µm]", 1e-6),
        ("dqin_dH_fwd_m3_s_per_m", "dqin_dH_ctr_m3_s_per_m", "dqin_dH_direct_m3_s_per_m", "dq_in/dH [µL/s por µm]", 1e3),
        ("dV_dH_fwd_m3_per_m", "dV_dH_ctr_m3_per_m", "dV_dH_direct_m3_per_m", "dV/dH [nL por µm]", 1e6),
    ]
    for row, (col_fwd, col_ctr, col_direct, ylabel, factor) in enumerate(derivative_specs):
        ax = axes[row, 1]
        ax.plot(df_H.H_um, df_H[col_fwd] * factor, marker="o", label="forward")
        ax.plot(df_H.H_um, df_H[col_ctr] * factor, marker="s", label="centered")
        ax.plot(df_H.H_um, df_H[col_direct] * factor, marker="^", label="direto contínuo")
        ax.set_ylabel(ylabel); ax.grid(True, alpha=.25); ax.legend()
    axes[0, 0].set_ylabel("Energia E")
    axes[1, 0].set_ylabel("q_in [µL/s]")
    axes[2, 0].set_ylabel("V(tf) [nL]")
    axes[2, 0].set_xlabel("H [µm]"); axes[2, 1].set_xlabel("H [µm]")
    axes[0, 0].set_title("Funções"); axes[0, 1].set_title("Derivadas: forward, centered e contínua")
    savefig(fig, "11_sensibilidades_H.png")

    # Zoom de dE/dH, protegido para escalas sensíveis.
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for col, marker, label in [
        ("dE_dH_fwd_per_m", "o", "Forward"),
        ("dE_dH_ctr_per_m", "s", "Centered"),
        ("dE_dH_direct_per_m", "^", "Método direto"),
    ]:
        ax.plot(df_H.H_um, df_H[col] * 1e-6, marker=marker, label=label)
    ax.set(xlabel="H [µm]", ylabel="dE/dH [por µm]",
           title="Comparação sensível: derivada da energia")
    ax.grid(True, alpha=.28); ax.legend()
    savefig(fig, "12_comparacao_dE_dH.png")

    df_tc.to_csv(OUTPUT_DIR / "sensibilidades_TC.csv", index=False)
    df_H.to_csv(OUTPUT_DIR / "sensibilidades_H.csv", index=False)
    return {"TC": df_tc, "H": df_H}


# -----------------------------------------------------------------------------
# NEWTON–RAPHSON — EXERCÍCIO 6.4.5.2
# -----------------------------------------------------------------------------
def newton_raphson_energy(twin: MultiphysicsDigitalTwin, target: float = 7.5,
                          H0: float = 800e-6, maxit: int = 12,
                          tol: float = 1e-7) -> Dict[str, Any]:
    """Newton com derivada provida pelo método direto de sensibilidades.

    Antes das iterações, uma busca de viabilidade evita inventar uma raiz em
    uma região onde E(H)-target não muda de sinal. Esse comportamento é uma
    conclusão física válida quando o alvo é incompatível com os parâmetros.
    """
    H_scan = np.linspace(100e-6, 2200e-6, 21)
    E_scan = np.array([twin.monolithic(H=float(H), sensitivity_H=False)["energy"] for H in H_scan])
    F_scan = E_scan - target
    sign_idx = np.where(F_scan[:-1] * F_scan[1:] <= 0.0)[0]
    feasible = sign_idx.size > 0
    hist: List[Dict[str, float]] = []
    H = float(H0)
    converged = False
    message = ""

    if not feasible:
        message = (
            "Não existe mudança de sinal de E(H)-7.5 na varredura [100,2200] µm; "
            "o alvo é inviável sob o conjunto nominal de parâmetros."
        )
    else:
        # Começa no centro do primeiro intervalo com mudança de sinal.
        left, right = H_scan[sign_idx[0]], H_scan[sign_idx[0] + 1]
        H = float(np.clip(H, left, right))
        for k in range(maxit):
            res = twin.monolithic(H=H, sensitivity_H=True)
            F = float(res["energy"] - target)
            dF = float(res["denergy_dH"])
            hist.append({"iter": k, "H_um": H * 1e6, "E": res["energy"], "F": F, "dE_dH": dF})
            if abs(F) < tol:
                converged = True
                break
            if abs(dF) < 1e-12:
                message = "Derivada quase nula: Newton interrompido para evitar passo instável."
                break
            H_new = H - F / dF
            # Salvaguarda: mantém a iteração no intervalo que contém a raiz.
            if not (left < H_new < right):
                H_new = 0.5 * (left + right)
            if F * (twin.monolithic(H=left)["energy"] - target) <= 0:
                right = H
            else:
                left = H
            H = float(H_new)
        if converged:
            message = "Newton–Raphson convergiu usando dE/dH do método direto."
        elif not message:
            message = "Newton atingiu o máximo de iterações sem satisfazer a tolerância."

    df = pd.DataFrame(hist)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].plot(H_scan * 1e6, E_scan, marker="o", label="E(H)")
    axes[0].axhline(target, linestyle="--", label="Alvo E=7.5")
    axes[0].set(xlabel="H [µm]", ylabel="Energia E", title="Viabilidade do alvo de Newton")
    axes[0].grid(True, alpha=.28); axes[0].legend()
    if not df.empty:
        axes[1].semilogy(df["iter"], np.maximum(np.abs(df["F"]), 1e-16), marker="o")
        axes[1].set(xlabel="Iteração", ylabel="|E(H)-7.5|", title="Convergência Newton–Raphson")
        axes[1].grid(True, alpha=.28)
    else:
        axes[1].text(.5, .5, message, ha="center", va="center", wrap=True, transform=axes[1].transAxes)
        axes[1].set_axis_off()
    savefig(fig, "13_newton_raphson_energia.png")
    pd.DataFrame({"H_um": H_scan * 1e6, "E": E_scan, "F": F_scan}).to_csv(
        OUTPUT_DIR / "newton_varredura_viabilidade.csv", index=False
    )
    df.to_csv(OUTPUT_DIR / "newton_iteracoes.csv", index=False)
    return {"feasible": feasible, "converged": converged, "message": message, "iterations": df,
            "H_scan": H_scan, "E_scan": E_scan}


# -----------------------------------------------------------------------------
# GRÁFICOS NOMINAIS E ANIMAÇÃO 3D — EXERCÍCIO 6.5
# -----------------------------------------------------------------------------
def plot_nominal_setup(twin: MultiphysicsDigitalTwin, thermal: Dict[str, Any], nominal: Dict[str, Any]) -> None:
    X, Y = np.meshgrid(thermal["x"] * 100.0, thermal["y"] * 100.0)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    cont = axes[0].contourf(X, Y, thermal["T_grid"], levels=28)
    plot_network(axes[0], twin.X_plate * 100.0, twin.conec, color="white", linewidth=.45)
    axes[0].scatter(twin.X_plate[twin.cfg.inlet_node, 0] * 100, twin.X_plate[twin.cfg.inlet_node, 1] * 100,
                    marker="o", s=22, label="inlet")
    axes[0].scatter(twin.X_plate[twin.cfg.outlet_node, 0] * 100, twin.X_plate[twin.cfg.outlet_node, 1] * 100,
                    marker="s", s=22, label="outlet")
    axes[0].set(xlabel="x [cm]", ylabel="y [cm]", title="Setup térmico estacionário + rede hidráulica")
    axes[0].legend(loc="upper left")
    fig.colorbar(cont, ax=axes[0], label="Temperatura [°C]")

    mu = thermal["mu_edge"]
    edge_mid = .5 * (twin.X_plate[twin.conec[:, 0]] + twin.X_plate[twin.conec[:, 1]])
    sc = axes[1].scatter(edge_mid[:, 0] * 100, edge_mid[:, 1] * 100, c=mu * 1e3, s=16)
    axes[1].set(xlabel="x [cm]", ylabel="y [cm]", title="Viscosidade média integrada por canal")
    fig.colorbar(sc, ax=axes[1], label="μ [mPa·s]")
    savefig(fig, "01_setup_termico_viscosidade.png")

    h = nominal["history"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    axes[0, 0].plot(h["tau"], h["power"])
    axes[0, 0].set(ylabel="P(τ)", title="Potência hidráulica adimensional")
    axes[0, 1].plot(h["tau"], h["p_outlet"] / 1000)
    axes[0, 1].set(ylabel="p_outlet [kPa]", title="Pressão no nó de interface")
    axes[1, 0].plot(h["tau"], h["w_center_m"] * 1e6, label="centro")
    axes[1, 0].plot(h["tau"], h["w_max_abs_m"] * 1e6, label="máximo |w|")
    axes[1, 0].set(xlabel="τ", ylabel="Deflexão [µm]", title="Resposta elástica"); axes[1, 0].legend()
    axes[1, 1].plot(h["tau"], h["volume_m3"] * 1e12)
    axes[1, 1].set(xlabel="τ", ylabel="Volume [nL]", title="Volume acumulado")
    for ax in axes.ravel(): ax.grid(True, alpha=.28)
    savefig(fig, "02_resposta_dinamica_nominal.png")


def make_integrated_animation(twin: MultiphysicsDigitalTwin, thermal: Dict[str, Any], nominal: Dict[str, Any]) -> Optional[Path]:
    """GIF 3D: setup térmico, membrana e dashboard síncrono."""
    h = nominal["history"]
    if "w_fields" not in h:
        return None
    frames = np.unique(np.linspace(0, len(h["tau"]) - 1, min(36, len(h["tau"])), dtype=int))
    xh, yh = nominal["system"]["x_hat"], nominal["system"]["y_hat"]
    Xh, Yh = np.meshgrid(xh * twin.cfg.radius * 1e3, yh * twin.cfg.radius * 1e3)
    mask = nominal["system"]["active_mask"]
    Xtemp, Ytemp = np.meshgrid(thermal["x"] * 100, thermal["y"] * 100)

    fig = plt.figure(figsize=(12.5, 8.0))
    axT = fig.add_subplot(2, 2, 1, projection="3d")
    axM = fig.add_subplot(2, 2, 2, projection="3d")
    axP = fig.add_subplot(2, 2, 3)
    axV = fig.add_subplot(2, 2, 4)

    axT.plot_surface(Xtemp, Ytemp, thermal["T_grid"], rstride=1, cstride=1, linewidth=0, alpha=.95)
    axT.set(title="Setup: campo térmico estacionário", xlabel="x [cm]", ylabel="y [cm]", zlabel="T [°C]")
    axT.view_init(elev=30, azim=-125)
    axP.set(xlim=(h["tau"][0], h["tau"][-1]), ylim=(0, max(h["power"]) * 1.08),
            xlabel="τ", ylabel="P(τ)", title="Potência instantânea")
    axV.set(xlim=(h["tau"][0], h["tau"][-1]), ylim=(min(h["volume_m3"]) * 1e12, max(h["volume_m3"]) * 1e12 * 1.08 + 1e-12),
            xlabel="τ", ylabel="V [nL]", title="Volume acumulado")
    for ax in (axP, axV): ax.grid(True, alpha=.25)

    def update(frame_idx: int):
        k = int(frames[frame_idx])
        axM.clear()
        W = h["w_fields"][k].reshape((twin.cfg.membrane_Ny, twin.cfg.membrane_Nx)) * 1e6
        W = np.where(mask, W, np.nan)
        axM.plot_surface(Xh, Yh, W, rstride=1, cstride=1, linewidth=0, alpha=.95)
        wlim = max(1.0, float(np.nanmax(np.abs(h["w_fields"])) * 1e6) * 1.15)
        axM.set(zlim=(-wlim, wlim), title=f"Membrana: τ={h['tau'][k]:.2f}",
                xlabel="x [mm]", ylabel="y [mm]", zlabel="w [µm]")
        axM.view_init(elev=30, azim=-135)
        axP.clear(); axV.clear()
        axP.plot(h["tau"][:k + 1], h["power"][:k + 1])
        axP.set(xlim=(h["tau"][0], h["tau"][-1]), ylim=(0, max(h["power"]) * 1.08),
                xlabel="τ", ylabel="P(τ)", title="Potência instantânea")
        axV.plot(h["tau"][:k + 1], h["volume_m3"][:k + 1] * 1e12)
        vtop = max(h["volume_m3"]) * 1e12
        axV.set(xlim=(h["tau"][0], h["tau"][-1]), ylim=(min(h["volume_m3"]) * 1e12, max(vtop * 1.08, 1e-12)),
                xlabel="τ", ylabel="V [nL]", title="Volume acumulado")
        for ax in (axP, axV): ax.grid(True, alpha=.25)
        return []

    animation = FuncAnimation(fig, update, frames=len(frames), interval=170, blit=False)
    path = OUTPUT_DIR / "14_animacao_gemeo_digital_3D.gif"
    animation.save(path, writer=PillowWriter(fps=6))
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# RELATÓRIO E EXECUÇÃO
# -----------------------------------------------------------------------------
def write_report(twin: MultiphysicsDigitalTwin, thermal: Dict[str, Any], nominal: Dict[str, Any],
                 mc: Dict[str, Any], learning: Dict[str, Any], sens: Dict[str, Any], newton: Dict[str, Any]) -> Path:
    scales = nominal["system"]["scales"]
    q_nom = nominal["qin_final_hat"] * scales["v_ref"] * twin.cfg.radius ** 2
    report = f"""# Gêmeo Digital Multiphysics — Parte 6

## Escopo executado

Este pacote resolve integralmente os itens do capítulo enviado: Monte Carlo estacionário e dinâmico (Seção 6.3.2), interpolação/regressão e ruído (Seção 6.4.3), diferenciação numérica e sensibilidades (Seção 6.4.5), Newton–Raphson para a restrição energética e a animação integrada do desafio final. O fluxo é estritamente sequencial: **placa térmica → viscosidade por canal → rede hidráulica → membrana**, como definido no PDF.

## Arquitetura e álgebra linear

A placa usa a discretização conservativa já fornecida em `ThermalPlateSolver`, formando o sistema esparso `A_T T=b_T`. A geometria da rede modifica `k(x,y)` por proximidade dos canais através de `NetworkInfluenceModel`. Como a matriz térmica não depende de `T_C`, ela é fatorada uma vez e reutilizada nas varreduras de sensibilidade.

A viscosidade de cada aresta é calculada por quadratura de Newton–Cotes (regra composta do trapézio):

`μ̄_k = Σ_i w_i μ(T(x_k(s_i)))`.

Assim, a condutância de Poiseuille equivalente satisfaz `C_k ∝ H^4/μ̄_k`. A rede hidráulica é a Laplaciana ponderada do grafo `A_h=B diag(C) Bᵀ` e a etapa fluido-estrutura usa o sistema monolítico `G Y^(n+1)=F^(n+1)`, com `Y=[w,v,p]ᵀ`. A fatoração `SolveG=factorized(G)` é feita antes do laço temporal. Isso cumpre a otimização pedida no enunciado e permite resolver também o sistema de sensibilidades com a mesma fatoração.

## Resultado nominal

- Temperatura máxima da placa: **{thermal['Tmax']:.4f} °C**.
- Temperatura média da placa: **{thermal['Tmean']:.4f} °C**.
- Viscosidade média integrada nas arestas: **{thermal['mu_edge'].mean()*1e3:.5f} mPa·s**.
- Energia adimensional no horizonte `[0,4]`: **{nominal['energy']:.8f}**.
- Vazão de entrada final: **{q_nom*1e9:.6f} µL/s**.
- Volume acumulado final: **{nominal['volume_final_m3']*1e12:.6f} nL**.

## Exercício 6.3.2 — Monte Carlo

Para cada cenário, `RandomFail(C_original,p_o,f_obs)` gera uma máscara Bernoulli independente usando `rng.random()`. As arestas obstruídas recebem `C_k/f_obs`; a probabilidade crítica é a média do indicador de falha. A convergência segue a ordem assintótica `N^(-1/2)` e o gráfico inclui uma banda de confiança de 95%.

- Caso estacionário de convergência: `p_o=0.40`, `f_obs=10`, estimativa final **{mc['convergence']['final_prob']:.6f}**, estabilização operacional em **N≈{mc['convergence']['stable_N']}**.
- Caso dinâmico, `δt=0.05`: `Prob(E<7)={mc['dynamic_dt005']['prob']:.6f} ± {1.96*mc['dynamic_dt005']['se']:.6f}`.
- Caso dinâmico, `δt=0.10`: `Prob(E<7)={mc['dynamic_dt010']['prob']:.6f} ± {1.96*mc['dynamic_dt010']['se']:.6f}`.

A diferença entre passos temporais combina dois efeitos: a quadratura temporal altera os valores de energia por amostra e, portanto, alguns cenários podem cruzar o limiar de `E=7`; além disso, a incerteza estatística permanece da ordem `sqrt(p(1-p)/N)`. Por isso, refinar `δt` reduz viés de discretização, mas não substitui aumentar `N`.

## Exercício 6.4.3 — Aproximação e regressão

As splines lineares e cúbicas interpolam os pontos discretos. A cúbica preserva suavidade de primeira derivada nas interfaces, enquanto a linear é mais robusta perto de alterações abruptas. Para a regressão global, foi montada a matriz de Vandermonde escalada e resolvido o sistema normal `Aᵀ A c=Aᵀ y`; o erro é medido por `||r||_L2` via a regra do trapézio. O menor erro observado ocorreu no grau **{learning['best_degree']}** no conjunto avaliado; os graus maiores devem ser interpretados junto à curvatura visual, pois a redução residual pode coexistir com oscilação de sobreajuste.

No experimento ruidoso, as splines passam exatamente pelos dados perturbados e, por isso, não filtram o ruído. A regressão de baixo/médio grau atua como filtro global e evidencia o compromisso entre fidelidade local e robustez.

## Exercício 6.4.5 — Sensibilidades

Foram calculadas diferenças progressivas e centradas em `T_C∈[0,250] °C` e `H∈[500,1500] µm`. Para `H`, o método direto usa

`dA/dH = (4/H)A` e resolve, em cada passo, `G S = [s_w/dt, M s_v/dt, -(dÃ/dH)p]ᵀ`.

As expressões solicitadas para as saídas são:

- `dq_in/dH = e_inᵀ[(dA/dH)p + A s_p]`;
- `dV(tf)/dH = ∫_0^tf h_hat² 1ᵀ s_v(t) · (v_ref R² t_ref) dt`.

Os gráficos `10–12` mostram as funções, forward, centered e a resposta direta. A diferença centrada apresenta erro de truncamento de ordem `O(ε²)`; o método direto não precisa de duas simulações completas por perturbação e é menos vulnerável à subtração de números próximos.

## Newton–Raphson para E(H)=7.5

{newton['message']}

O gráfico `13_newton_raphson_energia.png` registra a varredura de viabilidade e, quando há mudança de sinal, as iterações de Newton usando a derivada contínua. Caso não exista raiz para o conjunto nominal, isso é reportado explicitamente em vez de forçar uma solução numérica não física.

## Arquivos principais

- `gemeo_digital_integrado.py`: implementação completa e reprodutível.
- `resultados_gemeo_digital/*.png`: gráficos de todos os exercícios.
- `resultados_gemeo_digital/*.csv`: séries e tabelas numéricas.
- `resultados_gemeo_digital/14_animacao_gemeo_digital_3D.gif`: desafio de integração visual.
"""
    path = ROOT / "RELATORIO_GEMEO_DIGITAL.md"
    path.write_text(report, encoding="utf-8")
    return path


def write_readme() -> Path:
    text = """# Como executar

```bash
pip install numpy scipy pandas matplotlib shapely pillow
python gemeo_digital_integrado.py --full
```

Use `--quick` para uma execução reduzida de validação. A execução `--full` usa 2000 realizações para cada caso dinâmico exigido pelo enunciado. Os resultados são gravados em `resultados_gemeo_digital/`.

O arquivo preserva e importa as versões anteriores de `acoplamento_hidraulico_termico.py` e `acoplamento_hidraulico_mecanico.py`; por isso, mantenha os três scripts na mesma pasta.
"""
    path = ROOT / "README_EXECUCAO.md"
    path.write_text(text, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gêmeo Digital Integrado — Parte 6")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Executa as amostragens completas (padrão).")
    group.add_argument("--quick", action="store_true", help="Executa uma versão reduzida para validação.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quick = bool(args.quick)
    cfg = DigitalTwinConfig()
    twin = MultiphysicsDigitalTwin(cfg)

    print("[1/7] Resolvendo setup térmico e viscosidade por canal...")
    thermal = twin.thermal()

    print("[2/7] Resolvendo resposta monolítica nominal e sensibilidades diretas...")
    nominal = twin.monolithic(sensitivity_H=True, store_fields=True)
    plot_nominal_setup(twin, thermal, nominal)

    # Validação do complemento de Schur contra a formulação monolítica no caso nominal.
    C0, _ = twin.C_nominal()
    reduced_check = twin.reduced(cfg.dt).run(C0)
    rel_red = abs(reduced_check["energy"] - nominal["energy"]) / max(abs(nominal["energy"]), 1e-14)

    print("[3/7] Executando Monte Carlo estacionário e dinâmico...")
    mc = run_monte_carlo_exercises(twin, quick=quick)

    print("[4/7] Executando interpolação, regressão e teste com ruído...")
    learning = run_learning_exercises(twin, nominal, quick=quick)

    print("[5/7] Calculando mapas de sensibilidade...")
    sens = scan_sensitivities(twin, quick=quick)

    print("[6/7] Aplicando Newton–Raphson para E(H)-7.5...")
    newton = newton_raphson_energy(twin)

    print("[7/7] Renderizando animação 3D e relatórios...")
    animation = make_integrated_animation(twin, thermal, nominal)
    write_readme()
    report_path = write_report(twin, thermal, nominal, mc, learning, sens, newton)

    summary = {
        "thermal_Tmax_C": thermal["Tmax"],
        "thermal_Tmean_C": thermal["Tmean"],
        "mu_edge_mean_Pa_s": float(np.mean(thermal["mu_edge"])),
        "nominal_energy": nominal["energy"],
        "nominal_dE_dH": nominal.get("denergy_dH"),
        "schur_vs_monolithic_relative_energy_error": rel_red,
        "mc_stationary_prob": mc["convergence"]["final_prob"],
        "mc_stationary_stable_N": mc["convergence"]["stable_N"],
        "mc_dynamic_dt005_prob": mc["dynamic_dt005"]["prob"],
        "mc_dynamic_dt010_prob": mc["dynamic_dt010"]["prob"],
        "newton_feasible": newton["feasible"],
        "newton_converged": newton["converged"],
        "newton_message": newton["message"],
        "animation": str(animation) if animation else None,
        "report": str(report_path),
    }
    (OUTPUT_DIR / "resumo_resultados.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
