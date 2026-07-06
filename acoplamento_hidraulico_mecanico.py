# -*- coding: utf-8 -*-
"""
acoplamento_hidraulico_mecanico.py

Código para o Capítulo 5 - Acoplamento Hidráulico-Mecânico.
Foco: exercícios finais da seção 5.2.5 do PDF.

Visão Geral do Modelo Numérico:
Este script resolve um problema de interação fluido-estrutura (FSI) simplificado.
O modelo acopla:
1. Uma rede de canais microfluídicos (modelada como um grafo 1D com resistências hidráulicas).
2. Uma membrana elástica circular (modelada possivelmente por diferenças finitas ou elementos 
   finitos 2D, gerando matrizes de rigidez K e massa M).
O acoplamento se dá de forma monolítica através de uma matriz global em blocos.

Principais correções em relação ao código anterior:
1) O tempo usado nas rotinas é ADIMENSIONAL: tau = t/t_ref.
   Portanto dt_hat = 0.00625, 0.0125, 0.025, 0.05 e tau_final = 12.
2) O volume acumulado usa dt_físico = dt_hat*t_ref, não dt_hat diretamente.
3) A rotina 2 faz varredura dos parâmetros pedidos no PDF:
   H = 1000, 1250, 1500, 1750 um; p_inlet = 5e3, 1e4, 2e4 Pa;
   malhas 51x51 e 101x101; dt_hat = 0.00625, 0.0125, 0.025, 0.05.
4) As rotinas 4 e 5 usam o 3º modo calculado pela própria malha e monitoram
   o ponto de maior amplitude modal, evitando medir em nó quase nodal.
5) Os gráficos são reescalados automaticamente para unidades apresentáveis
   (um, kPa, uL/s, nL, mW/uW etc.).

Dependências:
    numpy, scipy, pandas, matplotlib, shapely

Uso básico:
    python acoplamento_hidraulico_mecanico.py --quick
    python acoplamento_hidraulico_mecanico.py --topics 1 2 3 4 5
    python acoplamento_hidraulico_mecanico.py --topics 2 --full-sweep

Observação sobre custo computacional:
    A varredura completa do item 2 envolve 4*2*3*4 = 96 simulações.
    Ela está implementada, mas pode demorar em máquinas comuns, especialmente
    nos casos 101x101. O sistema linear acoplado cresce rapidamente.
    Use --quick para gerar gráficos de apresentação rápidos.
"""

# Permite o uso de anotações de tipo modernas (ex: list[int] em vez de List[int]) 
# e avaliação postergada de anotações, essencial para clean code e type hinting.
from __future__ import annotations

# ============================================================
# IMPORTAÇÕES DA BIBLIOTECA PADRÃO
# ============================================================
import argparse          # Para criar a interface de linha de comando (CLI) e capturar argumentos (--topics, --quick).
import csv               # Para exportação eficiente dos históricos temporais (séries de tempo) em arquivos .csv.
import math              # Funções matemáticas escalares e constantes (como math.cos, math.sqrt).
import os                # Interações com o sistema operacional (embora pathlib seja preferido abaixo).
import time              # Usado para profiling básico (medir o tempo total de execução do script).
from dataclasses import dataclass, replace  # Para criar estruturas de dados imutáveis/configurações limpas (padrão DTO).
from pathlib import Path                   # Manipulação moderna e orientada a objetos de caminhos de arquivos/diretórios.
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple # Tipagem estática para documentar as assinaturas das funções.

# ============================================================
# IMPORTAÇÕES DE BIBLIOTECAS DE TERCEIROS (Cálculo e Dados)
# ============================================================
import numpy as np       # Base para todos os arrays N-dimensionais e operações algébricas vetorizadas.
import pandas as pd      # Usado aqui principalmente para manipulação tabular na construção do grafo (agregação de nós e arestas).

# ============================================================
# IMPORTAÇÕES DE VISUALIZAÇÃO
# ============================================================
import matplotlib
# Configura o backend do Matplotlib para "Agg" (Anti-Grain Geometry).
# IMPORTANTE: Isso desativa a interface gráfica interativa (GUI). É a melhor prática para
# scripts de simulação que geram e salvam muitas imagens em lote, evitando crashes de memória e janelas pipocando.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm # Colormaps para os gráficos de contorno (perfil de deflexão da membrana).

# ============================================================
# IMPORTAÇÕES DE GEOMETRIA COMPUTACIONAL (Grafo Fractal)
# ============================================================
# Shapely é usado aqui para lidar com a topologia da rede hidráulica.
# Ele facilita a união de segmentos de linha (canais) e a identificação de interseções/nós únicos.
from shapely.geometry import LineString
from shapely.ops import unary_union

# ============================================================
# IMPORTAÇÕES DE ÁLGEBRA LINEAR ESPARSA (O Core da Simulação)
# ============================================================
import scipy.sparse as sparse             # Para criar matrizes esparsas (CSR, CSC, COO, LIL). Essencial para K, M e matrizes do grafo.
import scipy.sparse.linalg as splinalg    # Solvers diretos (ex: splu) e iterativos para sistemas lineares esparsos.
from scipy.sparse.linalg import eigsh     # Eigenvalue Solver for Hermitian matrices. Usado para encontrar os modos de vibração naturais da membrana.

# ============================================================
# CONFIGURAÇÃO FÍSICA E NUMÉRICA
# ============================================================

# O uso de @dataclass(frozen=True) garante a imutabilidade do objeto de configuração.
# Isso é uma boa prática em simulações numéricas para evitar que parâmetros 
# sejam acidentalmente alterados no meio da execução, garantindo a reprodutibilidade.
@dataclass(frozen=True)
class Config:
    # --- Parâmetros da Rede Hidráulica ---
    levels: int = 3
    inlet_node: int = 0
    outlet_node: int = 5
    mu: float = 5e-4                         # Viscosidade dinâmica do fluido [Pa.s]
    channel_width: float = 1000e-6           # Largura (e profundidade assumida) dos canais: 1 mm [m]
    network_length_unit: float = 1e-3        # Fator de conversão: 1 unidade no grafo Shapely = 1 mm físico

    # --- Parâmetros da Membrana (Placa Circular Fina) ---
    radius: float = 0.0025                   # Raio R da membrana: 2.5 mm [m]
    thickness: float = 0.0001                # Espessura e da membrana: 0.1 mm [m]
    sigma: float = 200.0                     # Tensão superficial/tração inicial [N/m]
    rho: float = 900.0                       # Densidade do material [kg/m^3]
    beta_hat: float = 0.1                    # Coeficiente de amortecimento de Rayleigh (adimensional)
    Nx: int = 51                             # Resolução da malha espacial em X
    Ny: int = 51                             # Resolução da malha espacial em Y

    # --- Parâmetros de Tempo Adimensional ---
    # Intuição: Resolver equações diferenciais com parâmetros físicos que variam de 10^-6 a 10^3 
    # gera matrizes mal condicionadas (erros de arredondamento). O tempo adimensional tau = t/t_ref 
    # normaliza a evolução temporal.
    dt_hat: float = 0.025
    tau_final: float = 12.0

    # --- Parâmetros de Entrada (Condição de Contorno de Dirichlet) ---
    p_inlet: float = 5000.0                  # Pressão prescrita na entrada [Pa]

    # --- Parâmetros de Saída e Execução ---
    output_dir: str = "resultados_acoplamento"
    show_plots: bool = False

    # Amplitude inicial para a simulação de vibração livre (Ex. 4 e 5), normalizada por w_ref
    modal_initial_amplitude_hat: float = 0.5


def reference_scales(cfg: Config) -> Dict[str, float]:
    """
    Calcula as escalas características para adimensionalização do sistema FSI.
    
    A adimensionalização reduz o número de parâmetros livres e melhora a estabilidade numérica.
    Fundamentos:
    - L_ref: A escala espacial natural é o raio da membrana.
    - w_ref: Escala de deflexão assumida como pequena (1% do raio) para validar a teoria de pequenas deformações.
    - t_ref: Tempo característico de propagação de onda na membrana. Derivado da equação da onda: c = sqrt(sigma/(rho*e)). t = R/c.
    - p_ref: Escala de pressão derivada do balanço de forças estáticas na membrana: p ~ sigma * del^2(w) ~ sigma * w_ref / R^2.
    """
    R = cfg.radius
    w_ref = 0.01 * R
    t_ref = R * math.sqrt(cfg.rho * cfg.thickness / cfg.sigma)
    v_ref = w_ref / t_ref
    p_ref = cfg.sigma * w_ref / (R ** 2)
    return {"R": R, "w_ref": w_ref, "t_ref": t_ref, "v_ref": v_ref, "p_ref": p_ref}


# ============================================================
# GERAÇÃO DA REDE HIDRÁULICA E MATRIZ DE CONDUTÂNCIA
# ============================================================

def generate_graph_arrays(levels: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera as coordenadas dos nós e a matriz de conectividade (arestas) de um grafo fractal.
    
    A geometria é construída adicionando ramos recursivamente. O pacote Shapely é usado
    no final para fundir segmentos (unary_union) e extrair os nós topológicos únicos.
    Retorna:
        nodes_np: Coordenadas (x, y) de cada nó.
        edges_np: Lista de arestas [nó_origem, nó_destino].
    """
    nodes_data: List[Dict[str, float]] = []
    edges_raw: List[Tuple[int, int]] = []
    node_id = 0
    spine_length = 6

    # 1. Cria a "espinha dorsal" principal do canal
    spine_nodes: List[int] = []
    for i in range(spine_length):
        nodes_data.append({"x": i * 4.0, "y": 0.0})
        spine_nodes.append(node_id)
        if i > 0:
            edges_raw.append((node_id - 1, node_id))
        node_id += 1

    # 2. Função recursiva para gerar ramificações fractais (ex: rede vascular artificial)
    def add_fractal_branches(parent_id: int, px: float, py: float, angle: float, length: float, depth: int) -> None:
        nonlocal node_id
        if depth == 0:
            return
        angles = [angle + np.pi / 6, angle - np.pi / 6]
        branch_len = length * 0.75
        for a in angles:
            nx = px + branch_len * np.cos(a)
            ny = py + branch_len * np.sin(a)
            curr_id = node_id
            nodes_data.append({"x": float(nx), "y": float(ny)})
            edges_raw.append((parent_id, curr_id))
            node_id += 1
            add_fractal_branches(curr_id, nx, ny, a, branch_len, depth - 1)

    # Aplica ramificações acima e abaixo da espinha dorsal
    for s_id in spine_nodes[1:-1]:
        add_fractal_branches(s_id, nodes_data[s_id]["x"], nodes_data[s_id]["y"], np.pi / 2, 3.0, levels)
        add_fractal_branches(s_id, nodes_data[s_id]["x"], nodes_data[s_id]["y"], -np.pi / 2, 3.0, levels)

    # 3. Conecta as folhas soltas (leaf nodes) às bordas superior e inferior
    df_temp = pd.DataFrame(nodes_data)
    y_max = float(df_temp["y"].max() + 1.0)
    y_min = float(df_temp["y"].min() - 1.0)

    all_indices = [e[0] for e in edges_raw] + [e[1] for e in edges_raw]
    counts = pd.Series(all_indices).value_counts()
    leaf_ids = counts[counts == 1].index.tolist()
    leaf_ids = [idx for idx in leaf_ids if idx not in [spine_nodes[0], spine_nodes[-1]]]

    for l_id in leaf_ids:
        target_y = y_max if nodes_data[l_id]["y"] > 0 else y_min
        new_id = node_id
        nodes_data.append({"x": nodes_data[l_id]["x"], "y": target_y})
        edges_raw.append((l_id, new_id))
        node_id += 1

    # 4. Uso de geometria computacional (Shapely) para resolver interseções
    lines = [
        LineString([(nodes_data[e[0]]["x"], nodes_data[e[0]]["y"]),
                    (nodes_data[e[1]]["x"], nodes_data[e[1]]["y"])])
        for e in edges_raw
    ]

    df_nodes_final = pd.DataFrame(nodes_data)
    for y_lim in [y_max, y_min]:
        pts = df_nodes_final[df_nodes_final["y"] == y_lim].sort_values("x")
        if len(pts) > 1:
            lines.append(LineString(pts[["x", "y"]].values))

    merged_graph = unary_union(lines)
    final_nodes_map: Dict[Tuple[float, float], int] = {}
    final_nodes_list: List[List[float]] = []
    final_edges_list: List[List[int]] = []

    def get_node_id(pt: Sequence[float]) -> int:
        coords = (round(float(pt[0]), 6), round(float(pt[1]), 6))
        if coords not in final_nodes_map:
            final_nodes_map[coords] = len(final_nodes_list)
            final_nodes_list.append([float(pt[0]), float(pt[1])])
        return final_nodes_map[coords]

    segments = merged_graph.geoms if hasattr(merged_graph, "geoms") else [merged_graph]
    for seg in segments:
        id_start = get_node_id(seg.coords[0])
        id_end = get_node_id(seg.coords[-1])
        final_edges_list.append([id_start, id_end])

    nodes_np = np.asarray(final_nodes_list, dtype=float)
    edges_np = np.asarray(final_edges_list, dtype=int)
    edges_np = edges_np[edges_np[:, 0] != edges_np[:, 1]]
    return nodes_np, edges_np


def edge_lengths(Xno: np.ndarray, conec: np.ndarray, length_unit: float) -> np.ndarray:
    """Calcula a norma Euclidiana entre os nós de cada aresta para obter o comprimento físico L [m]."""
    return np.linalg.norm(Xno[conec[:, 1]] - Xno[conec[:, 0]], axis=1) * length_unit


def hydraulic_conductivities(Xno: np.ndarray, conec: np.ndarray, mu: float, width: float, length_unit: float) -> Dict[str, np.ndarray]:
    """
    Calcula a condutância hidráulica C de cada canal assumindo fluxo laminar incompressível (Poiseuille).
    
    Trade-off de modelagem:
    A seção transversal real assumida é quadrada (width x width). A solução analítica exata para o fluxo
    de Poiseuille em dutos retangulares exige uma série infinita. 
    Para simplificar o custo computacional, usa-se a aproximação pelo Diâmetro Hidráulico Equivalente (D_eq).
    D_eq = sqrt(4 * Area / pi).
    Isso permite usar a equação clássica de tubos circulares: Condutância C = (pi * D_eq^4) / (128 * mu * L).
    """
    L = edge_lengths(Xno, conec, length_unit)
    area_edge = np.full(conec.shape[0], width * width, dtype=float)
    D_eq = np.sqrt(4.0 * area_edge / np.pi)
    
    # Termo geométrico/viscoso da lei de Poiseuille (Q = kappa * delta_P / L)
    kappa = np.pi * D_eq ** 4 / (128.0 * mu)
    C = kappa / L
    return {"lengths": L, "conductance_edge": C, "D_eq": D_eq, "area": area_edge}


def assembly_hydraulic(conec: np.ndarray, conductance_edge: np.ndarray) -> sparse.csr_matrix:
    """
    Monta a matriz esparsa global do sistema hidráulico baseada nas leis de Kirchhoff.
    
    Analogia elétrica:
    - Tensão -> Pressão P
    - Corrente -> Vazão Q
    - Resistência -> 1/C
    
    A conservação de massa em cada nó exige que o somatório das vazões seja zero.
    Gera a matriz Laplaciana do grafo ponderada pelas condutâncias.
    - Diagonal principal [i, i]: soma das condutâncias conectadas ao nó i.
    - Fora da diagonal [i, j]: negativo da condutância entre os nós i e j (se existir conexão).
    Retorna no formato CSR (Compressed Sparse Row) otimizado para operações matriciais vetorizadas rápidas.
    """
    nv = int(np.max(conec)) + 1
    rows, cols, data = [], [], []
    diag = np.zeros(nv, dtype=float)
    for k, (i_raw, j_raw) in enumerate(conec):
        i, j = int(i_raw), int(j_raw)
        c = float(conductance_edge[k])
        # Adiciona a condutância à diagonal dos nós conectados
        diag[i] += c
        diag[j] += c
        
        # Preenche os termos cruzados (simétricos)
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([-c, -c])
        
    rows.extend(range(nv))
    cols.extend(range(nv))
    data.extend(diag.tolist())
    
    # Monta usando COO form primeiro (rápido para construir) e converte para CSR (rápido para álgebra)
    return sparse.coo_matrix((data, (rows, cols)), shape=(nv, nv)).tocsr()


def impose_pressure_bc_on_A(A: sparse.csr_matrix, node: int) -> sparse.csr_matrix:
    """
    Impõe Condição de Contorno (BC) de Dirichlet (pressão prescrita) no nó especificado.
    
    Técnica de Penalização / Substituição de Linha:
    Substitui a linha correspondente ao nó na matriz de condutância por uma identidade:
    A[node, :] = 0 e A[node, node] = 1.
    Durante o 'solve', o vetor RHS (Right-Hand Side) receberá o valor da pressão na posição 'node', 
    forçando a solução exata p[node] = p_prescrita.
    """
    # Converte para LIL (List of Lists), que é mais eficiente para modificar a estrutura de esparsidade
    A = A.tolil(copy=True)
    A[node, :] = 0.0
    A[node, node] = 1.0
    return A.tocsr()

# ============================================================
# MEMBRANA ELÁSTICA
# ============================================================

def ij2n(i: int, j: int, Nx: int) -> int:
    """
    Mapeamento bidimensional para unidimensional.
    Intuição: Para resolver o sistema linear na forma matricial A*x = b, a matriz 
    bidimensional de deflexões (w) precisa ser "achatada" (flattened) em um vetor coluna 1D.
    """
    return i + j * Nx

def n2ij(n: int, Nx: int) -> Tuple[int, int]:
    """Operação inversa de ij2n. Útil para pós-processamento e visualização."""
    return n % Nx, n // Nx

def is_inside_unit_circle(x_hat: float, y_hat: float) -> bool:
    """Verifica se uma coordenada adimensional está dentro da membrana circular (raio=1.0)."""
    return x_hat * x_hat + y_hat * y_hat <= 1.0

def is_restricted_point(i: int, j: int, Nx: int, Ny: int, x_hat: np.ndarray, y_hat: np.ndarray, use_circular_mask: bool = True) -> bool:
    """
    Define os nós engastados (onde a deflexão w = 0 é imposta).
    
    Regras de restrição:
    1. As bordas do domínio retangular computacional são sempre restritas.
    2. Se a máscara circular estiver ativa, qualquer nó fora do raio unitário é restrito.
    3. Trade-off (estabilidade x precisão do contorno): A verificação de vizinhos (x_hat[i+1], etc.)
       garante que a "fronteira em degrau" gerada pela malha cartesiana não tenha nós soltos (flutuantes)
       que poderiam deixar a matriz de rigidez singular.
    """
    if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
        return True
    if not use_circular_mask:
        return False
    xh, yh = x_hat[i], y_hat[j]
    if not is_inside_unit_circle(xh, yh):
        return True
    
    # Exige que todos os 4 vizinhos imediatos também estejam no domínio. 
    # Isso cria uma "margem de segurança" numérica na borda recortada.
    for xn, yn in [(x_hat[i + 1], yh), (x_hat[i - 1], yh), (xh, y_hat[j + 1]), (xh, y_hat[j - 1])]:
        if not is_inside_unit_circle(xn, yn):
            return True
    return False

def build_active_mask(Nx: int, Ny: int, x_hat: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """Cria uma matriz booleana indicando os graus de liberdade (nós móveis)."""
    mask = np.zeros((Ny, Nx), dtype=bool)
    for j in range(Ny):
        for i in range(Nx):
            mask[j, i] = not is_restricted_point(i, j, Nx, Ny, x_hat, y_hat, True)
    return mask

def assembly_membrane(Nx: int, Ny: int, Lx_hat: float = 2.0, Ly_hat: float = 2.0, big_number: float = 1e8) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Monta as matrizes de Rigidez Adimensional (K) e Massa Adimensional (M).
    
    Fundamentos:
    Aproxima o operador espacial Laplaciano negativo ($-\nabla^2 w$) usando o estêncil de 
    Diferenças Finitas de 5 pontos (ordem $O(h^2)$):
    Laplaciano_ij ≈ (w_{i-1,j} + w_{i+1,j} + w_{i,j-1} + w_{i,j+1} - 4w_{i,j}) / h^2
    
    Condições de Contorno:
    Usa o método da penalização ("big_number" na diagonal) para forçar que os nós fora do 
    domínio ativo tenham deslocamento tendendo a zero de forma rigorosa sem quebrar a simetria de K.
    """
    x_hat = np.linspace(-Lx_hat / 2.0, Lx_hat / 2.0, Nx)
    y_hat = np.linspace(-Ly_hat / 2.0, Ly_hat / 2.0, Ny)
    h_hat = Lx_hat / (Nx - 1)
    nunk = Nx * Ny

    rows, cols, data = [], [], []
    mass_diag = np.zeros(nunk, dtype=float)
    stiffness_scale = 1.0 / (h_hat ** 2)

    active_mask = build_active_mask(Nx, Ny, x_hat, y_hat)
    for j in range(Ny):
        for i in range(Nx):
            Ic = ij2n(i, j, Nx)
            mass_diag[Ic] = 1.0 # A massa concentrada ("lumped mass") forma uma matriz diagonal.
            
            if not active_mask[j, i]:
                # Penalização para nós fixos: K_ii * w_i = Força. Com K_ii muito grande, w_i ~ 0.
                rows.append(Ic); cols.append(Ic); data.append(big_number)
                continue
                
            Ie, Iw = ij2n(i + 1, j, Nx), ij2n(i - 1, j, Nx)
            In, Is = ij2n(i, j + 1, Nx), ij2n(i, j - 1, Nx)
            
            # Matriz Laplaciana negativa (garante que K seja definida positiva)
            rows.extend([Ic, Ic, Ic, Ic, Ic])
            cols.extend([Ic, Ie, Iw, In, Is])
            data.extend([4.0 * stiffness_scale, -stiffness_scale, -stiffness_scale, -stiffness_scale, -stiffness_scale])

    K = sparse.coo_matrix((data, (rows, cols)), shape=(nunk, nunk)).tocsr()
    M = sparse.diags(mass_diag, format="csr")
    return K, M, x_hat, y_hat, h_hat, active_mask

def solve_membrane_modes(K: sparse.csr_matrix, M: sparse.csr_matrix, num_modes: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve o problema de autovalor generalizado do sistema não amortecido: K * phi = lambda * M * phi.
    
    A solução revela os modos naturais de vibração (autovetores phi) e as frequências 
    naturais ao quadrado (autovalores lambda = omega_hat^2).
    
    Limitação de design tratada:
    Em vibrações estruturais, os autovetores podem ter sinais arbitrários devido à natureza 
    dos solvers iterativos (subespaço de Krylov). O código aplica uma normalização 
    orientada ao maior deslocamento absoluto para garantir que o pico modal comece positivo.
    Isso padroniza a fase da oscilação para as rotinas 4 e 5.
    """
    n = K.shape[0]
    k = min(num_modes, n - 2)
    
    # eigsh: Eigenvalue Solver for Hermitian matrices da biblioteca ARPACK.
    # which="SM" (Smallest Magnitude) procura os menores autovalores, correspondentes
    # às frequências naturais de base que dominam a resposta dinâmica.
    evals, evecs = eigsh(K, k=k, M=M, which="SM", v0=np.ones(n))
    
    # Ordenação crescente das frequências naturais
    idx = np.argsort(evals)
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])
    
    # Garantia de positividade do deslocamento máximo no autovetor (fase consistente)
    for col in range(evecs.shape[1]):
        m = np.argmax(np.abs(evecs[:, col]))
        if evecs[m, col] < 0:
            evecs[:, col] *= -1.0
            
    # Frequência angular natural adimensional omega_hat = sqrt(lambda)
    omega = np.sqrt(np.maximum(evals, 0.0))
    return evals, evecs, omega

# ============================================================
# MEMBRANA ELÁSTICA
# ============================================================

def ij2n(i: int, j: int, Nx: int) -> int:
    """
    Mapeamento bidimensional para unidimensional.
    Intuição: Para resolver o sistema linear na forma matricial A*x = b, a matriz 
    bidimensional de deflexões (w) precisa ser "achatada" (flattened) em um vetor coluna 1D.
    """
    return i + j * Nx

def n2ij(n: int, Nx: int) -> Tuple[int, int]:
    """Operação inversa de ij2n. Útil para pós-processamento e visualização."""
    return n % Nx, n // Nx

def is_inside_unit_circle(x_hat: float, y_hat: float) -> bool:
    """Verifica se uma coordenada adimensional está dentro da membrana circular (raio=1.0)."""
    return x_hat * x_hat + y_hat * y_hat <= 1.0

def is_restricted_point(i: int, j: int, Nx: int, Ny: int, x_hat: np.ndarray, y_hat: np.ndarray, use_circular_mask: bool = True) -> bool:
    """
    Define os nós engastados (onde a deflexão w = 0 é imposta).
    
    Regras de restrição:
    1. As bordas do domínio retangular computacional são sempre restritas.
    2. Se a máscara circular estiver ativa, qualquer nó fora do raio unitário é restrito.
    3. Trade-off (estabilidade x precisão do contorno): A verificação de vizinhos (x_hat[i+1], etc.)
       garante que a "fronteira em degrau" gerada pela malha cartesiana não tenha nós soltos (flutuantes)
       que poderiam deixar a matriz de rigidez singular.
    """
    if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
        return True
    if not use_circular_mask:
        return False
    xh, yh = x_hat[i], y_hat[j]
    if not is_inside_unit_circle(xh, yh):
        return True
    
    # Exige que todos os 4 vizinhos imediatos também estejam no domínio. 
    # Isso cria uma "margem de segurança" numérica na borda recortada.
    for xn, yn in [(x_hat[i + 1], yh), (x_hat[i - 1], yh), (xh, y_hat[j + 1]), (xh, y_hat[j - 1])]:
        if not is_inside_unit_circle(xn, yn):
            return True
    return False

def build_active_mask(Nx: int, Ny: int, x_hat: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """Cria uma matriz booleana indicando os graus de liberdade (nós móveis)."""
    mask = np.zeros((Ny, Nx), dtype=bool)
    for j in range(Ny):
        for i in range(Nx):
            mask[j, i] = not is_restricted_point(i, j, Nx, Ny, x_hat, y_hat, True)
    return mask

def assembly_membrane(Nx: int, Ny: int, Lx_hat: float = 2.0, Ly_hat: float = 2.0, big_number: float = 1e8) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Monta as matrizes de Rigidez Adimensional (K) e Massa Adimensional (M).
    
    Fundamentos:
    Aproxima o operador espacial Laplaciano negativo ($-\nabla^2 w$) usando o estêncil de 
    Diferenças Finitas de 5 pontos (ordem $O(h^2)$):
    Laplaciano_ij ≈ (w_{i-1,j} + w_{i+1,j} + w_{i,j-1} + w_{i,j+1} - 4w_{i,j}) / h^2
    
    Condições de Contorno:
    Usa o método da penalização ("big_number" na diagonal) para forçar que os nós fora do 
    domínio ativo tenham deslocamento tendendo a zero de forma rigorosa sem quebrar a simetria de K.
    """
    x_hat = np.linspace(-Lx_hat / 2.0, Lx_hat / 2.0, Nx)
    y_hat = np.linspace(-Ly_hat / 2.0, Ly_hat / 2.0, Ny)
    h_hat = Lx_hat / (Nx - 1)
    nunk = Nx * Ny

    rows, cols, data = [], [], []
    mass_diag = np.zeros(nunk, dtype=float)
    stiffness_scale = 1.0 / (h_hat ** 2)

    active_mask = build_active_mask(Nx, Ny, x_hat, y_hat)
    for j in range(Ny):
        for i in range(Nx):
            Ic = ij2n(i, j, Nx)
            mass_diag[Ic] = 1.0 # A massa concentrada ("lumped mass") forma uma matriz diagonal.
            
            if not active_mask[j, i]:
                # Penalização para nós fixos: K_ii * w_i = Força. Com K_ii muito grande, w_i ~ 0.
                rows.append(Ic); cols.append(Ic); data.append(big_number)
                continue
                
            Ie, Iw = ij2n(i + 1, j, Nx), ij2n(i - 1, j, Nx)
            In, Is = ij2n(i, j + 1, Nx), ij2n(i, j - 1, Nx)
            
            # Matriz Laplaciana negativa (garante que K seja definida positiva)
            rows.extend([Ic, Ic, Ic, Ic, Ic])
            cols.extend([Ic, Ie, Iw, In, Is])
            data.extend([4.0 * stiffness_scale, -stiffness_scale, -stiffness_scale, -stiffness_scale, -stiffness_scale])

    K = sparse.coo_matrix((data, (rows, cols)), shape=(nunk, nunk)).tocsr()
    M = sparse.diags(mass_diag, format="csr")
    return K, M, x_hat, y_hat, h_hat, active_mask

def solve_membrane_modes(K: sparse.csr_matrix, M: sparse.csr_matrix, num_modes: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve o problema de autovalor generalizado do sistema não amortecido: K * phi = lambda * M * phi.
    
    A solução revela os modos naturais de vibração (autovetores phi) e as frequências 
    naturais ao quadrado (autovalores lambda = omega_hat^2).
    
    Limitação de design tratada:
    Em vibrações estruturais, os autovetores podem ter sinais arbitrários devido à natureza 
    dos solvers iterativos (subespaço de Krylov). O código aplica uma normalização 
    orientada ao maior deslocamento absoluto para garantir que o pico modal comece positivo.
    Isso padroniza a fase da oscilação para as rotinas 4 e 5.
    """
    n = K.shape[0]
    k = min(num_modes, n - 2)
    
    # eigsh: Eigenvalue Solver for Hermitian matrices da biblioteca ARPACK.
    # which="SM" (Smallest Magnitude) procura os menores autovalores, correspondentes
    # às frequências naturais de base que dominam a resposta dinâmica.
    evals, evecs = eigsh(K, k=k, M=M, which="SM", v0=np.ones(n))
    
    # Ordenação crescente das frequências naturais
    idx = np.argsort(evals)
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])
    
    # Garantia de positividade do deslocamento máximo no autovetor (fase consistente)
    for col in range(evecs.shape[1]):
        m = np.argmax(np.abs(evecs[:, col]))
        if evecs[m, col] < 0:
            evecs[:, col] *= -1.0
            
    # Frequência angular natural adimensional omega_hat = sqrt(lambda)
    omega = np.sqrt(np.maximum(evals, 0.0))
    return evals, evecs, omega

# ============================================================
# ACOPLAMENTO E SOLUÇÃO TEMPORAL
# ============================================================

def build_U(Nx: int, Ny: int, active_mask: np.ndarray, n_p: int, outlet_node: int) -> sparse.csr_matrix:
    """
    Constrói a matriz de acoplamento U, que faz a ponte cinemática entre a malha 2D e o grafo 1D.
    
    A física do problema exige que a vazão de fluido empurrada pela membrana (integral da velocidade v 
    na área ativa) entre no nó de descarga da rede hidráulica (outlet_node).
    Matematicamente: q_outlet = integral(v) dA. 
    A matriz U possui dimensão [Nós_hidráulicos x Nós_mecânicos]. Ela é esparsa e contém 1s na 
    linha do 'outlet_node' nas colunas correspondentes aos nós ativos da membrana.
    """
    n_m = Nx * Ny
    # LIL é usado pois facilita a atribuição rápida por linhas (slices).
    U = sparse.lil_matrix((n_p, n_m), dtype=float)
    active_flat = active_mask.ravel(order="C")
    U[outlet_node, np.where(active_flat)[0]] = 1.0
    return U.tocsr()

def prepare_system(cfg: Config) -> Dict[str, object]:
    """
    Monta a matriz global monolítica do sistema e prepara a fatoração LU.
    
    Formulação:
    O sistema contínuo é discretizado no tempo usando Euler Implícito (ordem 1, incondicionalmente estável),
    necessário para lidar com a rigidez numérica comum em problemas FSI.
    O vetor de incógnitas é [w, v, p]^T.
    
    A matriz global Aglob é formada por blocos:
    1. Cinemática (dv/dt = w):         (1/dt)*I * w - I * v = (1/dt)*w_ant
    2. Eq. Movimento Membrana:         K * w + [(1/dt)*M + D] * v - U^T * p = (1/dt)*M * v_ant
    3. Conservação Massa Hidráulica:   (h^2)*U * v + A_hat * p = Termos_de_Fronteira (p_inlet)
    
    Decisão de Projeto:
    Optou-se por um solver Monolítico (todas equações acopladas numa única matriz) em vez de Particionado
    (resolver fluido, depois estrutura, e iterar).
    - Trade-off: Monolítico é robusto e incondicionalmente estável para acoplamentos fortes, 
      mas consome mais memória (Aglob é maior).
    """
    Xno, conec = generate_graph_arrays(cfg.levels)
    hyd = hydraulic_conductivities(Xno, conec, cfg.mu, cfg.channel_width, cfg.network_length_unit)
    A_phys = assembly_hydraulic(conec, hyd["conductance_edge"])

    K, M, x_hat, y_hat, h_hat, active_mask = assembly_membrane(cfg.Nx, cfg.Ny)
    n_p, n_m = Xno.shape[0], cfg.Nx * cfg.Ny
    U = build_U(cfg.Nx, cfg.Ny, active_mask, n_p, cfg.outlet_node)

    scales = reference_scales(cfg)
    
    # Adimensionalização da matriz hidráulica para garantir ordem de grandeza similar 
    # às matrizes estruturais, mitigando problemas de condicionamento na inversão.
    A_hat = A_phys * scales["p_ref"] / (scales["v_ref"] * cfg.radius ** 2)
    A_hat = impose_pressure_bc_on_A(A_hat, cfg.inlet_node)

    I = sparse.identity(n_m, format="csr")
    D = cfg.beta_hat * M # Amortecimento proporcional à massa (Rayleigh simplificado)
    dt = cfg.dt_hat
    
    # Montagem da matriz em blocos 3x3 usando lista de listas
    blocks = [
        [(1.0 / dt) * I, -I, None],
        [K, (1.0 / dt) * M + D, -U.T],
        [None, (h_hat ** 2) * U, A_hat],
    ]
    # CSC é o formato ótimo para o solver SuperLU
    Aglob = sparse.bmat(blocks, format="csc")
    
    # Pré-fatoração LU. Como Aglob não muda no tempo, a fatoração cara O(N^3) ocorre apenas uma vez.
    # O loop temporal resolverá apenas substituições retroativas O(N^2), barateando o custo global.
    solver = splinalg.splu(Aglob)

    return {
        "Xno": Xno,
        "conec": conec,
        "hyd": hyd,
        "A_phys": A_phys,
        "A_hat": A_hat,
        "K": K,
        "M": M,
        "U": U,
        "x_hat": x_hat,
        "y_hat": y_hat,
        "h_hat": h_hat,
        "active_mask": active_mask,
        "solver": solver,
        "scales": scales,
        "n_p": n_p,
        "n_m": n_m,
    }

ForcingFunc = Callable[[float, float, Dict[str, float]], float]

def run_simulation(
    cfg: Config,
    initial_state: Optional[Dict[str, np.ndarray]] = None,
    forcing_func: Optional[ForcingFunc] = None,
    monitor_idx: Optional[int] = None,
    snapshot_tau: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    """
    Itera sobre o domínio temporal resolvendo o sistema linear em cada passo.
    
    O algoritmo implementa uma marcha no tempo.
    O vetor do lado direito (RHS) é atualizado em cada iteração contendo os termos de memória 
    (w e v do passo anterior) e os carregamentos externos (pressão no inlet).
    """
    system = prepare_system(cfg)
    n_m, n_p = int(system["n_m"]), int(system["n_p"])
    M, U = system["M"], system["U"]
    solver = system["solver"]
    h_hat = float(system["h_hat"])
    scales = system["scales"]

    # Criação do vetor de tempo. A adição de 0.5 * dt_hat lida com imprecisões de float no arange.
    tau = np.arange(0.0, cfg.tau_final + 0.5 * cfg.dt_hat, cfg.dt_hat)
    t_phys = tau * scales["t_ref"]
    w = np.zeros(n_m, dtype=float)
    v = np.zeros(n_m, dtype=float)
    
    if initial_state is not None:
        w[:] = initial_state.get("w", w)
        v[:] = initial_state.get("v", v)

    if monitor_idx is None:
        monitor_idx = ij2n(cfg.Nx // 2, cfg.Ny // 2, cfg.Nx)

    active_indices = U.getrow(cfg.outlet_node).indices
    vol = 0.0

    # Pré-alocação de memória para os vetores de histórico (melhor performance que o método .append() em listas)
    hist = {
        "tau": tau,
        "t_phys": t_phys,
        "p_inlet": np.zeros_like(tau),
        "p_outlet": np.zeros_like(tau),
        "q_outlet": np.zeros_like(tau),
        "w_center": np.zeros_like(tau),
        "w_monitor": np.zeros_like(tau),
        "volume": np.zeros_like(tau),
        "power": np.zeros_like(tau),
        "w_max_abs": np.zeros_like(tau),
    }
    
    snapshots: Dict[float, np.ndarray] = {}
    snapshot_tau = [] if snapshot_tau is None else list(snapshot_tau)
    snapshot_taken = {float(s): False for s in snapshot_tau}

    last_p = np.zeros(n_p, dtype=float)
    
    # Loop de Marcha no Tempo
    for k, tau_k in enumerate(tau):
        # Constrói vetor B do sistema A*x = B
        rhs = np.zeros(2 * n_m + n_p, dtype=float)
        rhs[:n_m] = w / cfg.dt_hat               # Inércia cinemática
        rhs[n_m:2 * n_m] = M.dot(v) / cfg.dt_hat # Momento linear anterior

        # Atualiza o forçamento (se houver, como p_inlet caindo a zero ou harmônico)
        p_in = forcing_func(float(tau_k), float(t_phys[k]), scales) if forcing_func else cfg.p_inlet
        rhs[2 * n_m + cfg.inlet_node] = p_in / scales["p_ref"] # Aplica BC de Dirichlet no nó da matriz penalizado

        # Solve via substituição retroativa (Forward/Backward substitution da LU)
        sol = solver.solve(rhs)
        
        # Desempacota o vetor solução (unpacking)
        w = sol[:n_m]
        v = sol[n_m:2 * n_m]
        p = sol[2 * n_m:]
        last_p = p.copy()

        # Pós-processamento e redimensionalização (volta ao domínio físico)
        p_out = p[cfg.outlet_node] * scales["p_ref"]
        
        # Integração numérica da vazão usando regra do retângulo: sum(v) * dA (sendo dA = h_hat^2 * R^2)
        q_out = (h_hat ** 2) * np.sum(v[active_indices]) * scales["v_ref"] * cfg.radius ** 2
        
        # Integração temporal explícita para o volume acumulado
        vol += q_out * cfg.dt_hat * scales["t_ref"]

        # Armazenamento de variáveis escalares (probe points)
        hist["p_inlet"][k] = p_in
        hist["p_outlet"][k] = p_out
        hist["q_outlet"][k] = q_out
        hist["w_center"][k] = w[ij2n(cfg.Nx // 2, cfg.Ny // 2, cfg.Nx)] * scales["w_ref"]
        hist["w_monitor"][k] = w[monitor_idx] * scales["w_ref"]
        hist["volume"][k] = vol
        hist["power"][k] = p_in * q_out  # Potência P = p * Q
        hist["w_max_abs"][k] = np.max(np.abs(w)) * scales["w_ref"]

        # Captura de campo completo (snapshots) em tempos específicos.
        # Risco: Salvar arrays de campo completo a cada passo de tempo esgotaria a RAM. 
        # A escolha de um array de tempos específicos contorna isso.
        for s in snapshot_tau:
            s = float(s)
            if not snapshot_taken[s] and tau_k >= s:
                snapshots[s] = w.copy() * scales["w_ref"]
                snapshot_taken[s] = True

    state = {"w": w.copy(), "v": v.copy(), "p": last_p.copy(), "snapshots": snapshots}
    return hist, state, system

# ============================================================
# ACOPLAMENTO E SOLUÇÃO TEMPORAL
# ============================================================

def build_U(Nx: int, Ny: int, active_mask: np.ndarray, n_p: int, outlet_node: int) -> sparse.csr_matrix:
    """
    Constrói a matriz de acoplamento U, que faz a ponte cinemática entre a malha 2D e o grafo 1D.
    
    A física do problema exige que a vazão de fluido empurrada pela membrana (integral da velocidade v 
    na área ativa) entre no nó de descarga da rede hidráulica (outlet_node).
    Matematicamente: q_outlet = integral(v) dA. 
    A matriz U possui dimensão [Nós_hidráulicos x Nós_mecânicos]. Ela é esparsa e contém 1s na 
    linha do 'outlet_node' nas colunas correspondentes aos nós ativos da membrana.
    """
    n_m = Nx * Ny
    # LIL é usado pois facilita a atribuição rápida por linhas (slices).
    U = sparse.lil_matrix((n_p, n_m), dtype=float)
    active_flat = active_mask.ravel(order="C")
    U[outlet_node, np.where(active_flat)[0]] = 1.0
    return U.tocsr()

def prepare_system(cfg: Config) -> Dict[str, object]:
    """
    Monta a matriz global monolítica do sistema e prepara a fatoração LU.
    
    Formulação:
    O sistema contínuo é discretizado no tempo usando Euler Implícito (ordem 1, incondicionalmente estável),
    necessário para lidar com a rigidez numérica comum em problemas FSI.
    O vetor de incógnitas é [w, v, p]^T.
    
    A matriz global Aglob é formada por blocos:
    1. Cinemática (dv/dt = w):         (1/dt)*I * w - I * v = (1/dt)*w_ant
    2. Eq. Movimento Membrana:         K * w + [(1/dt)*M + D] * v - U^T * p = (1/dt)*M * v_ant
    3. Conservação Massa Hidráulica:   (h^2)*U * v + A_hat * p = Termos_de_Fronteira (p_inlet)
    
    Decisão de Projeto:
    Optou-se por um solver Monolítico (todas equações acopladas numa única matriz) em vez de Particionado
    (resolver fluido, depois estrutura, e iterar).
    - Trade-off: Monolítico é robusto e incondicionalmente estável para acoplamentos fortes, 
      mas consome mais memória (Aglob é maior).
    """
    Xno, conec = generate_graph_arrays(cfg.levels)
    hyd = hydraulic_conductivities(Xno, conec, cfg.mu, cfg.channel_width, cfg.network_length_unit)
    A_phys = assembly_hydraulic(conec, hyd["conductance_edge"])

    K, M, x_hat, y_hat, h_hat, active_mask = assembly_membrane(cfg.Nx, cfg.Ny)
    n_p, n_m = Xno.shape[0], cfg.Nx * cfg.Ny
    U = build_U(cfg.Nx, cfg.Ny, active_mask, n_p, cfg.outlet_node)

    scales = reference_scales(cfg)
    
    # Adimensionalização da matriz hidráulica para garantir ordem de grandeza similar 
    # às matrizes estruturais, mitigando problemas de condicionamento na inversão.
    A_hat = A_phys * scales["p_ref"] / (scales["v_ref"] * cfg.radius ** 2)
    A_hat = impose_pressure_bc_on_A(A_hat, cfg.inlet_node)

    I = sparse.identity(n_m, format="csr")
    D = cfg.beta_hat * M # Amortecimento proporcional à massa (Rayleigh simplificado)
    dt = cfg.dt_hat
    
    # Montagem da matriz em blocos 3x3 usando lista de listas
    blocks = [
        [(1.0 / dt) * I, -I, None],
        [K, (1.0 / dt) * M + D, -U.T],
        [None, (h_hat ** 2) * U, A_hat],
    ]
    # CSC é o formato ótimo para o solver SuperLU
    Aglob = sparse.bmat(blocks, format="csc")
    
    # Pré-fatoração LU. Como Aglob não muda no tempo, a fatoração cara O(N^3) ocorre apenas uma vez.
    # O loop temporal resolverá apenas substituições retroativas O(N^2), barateando o custo global.
    solver = splinalg.splu(Aglob)

    return {
        "Xno": Xno,
        "conec": conec,
        "hyd": hyd,
        "A_phys": A_phys,
        "A_hat": A_hat,
        "K": K,
        "M": M,
        "U": U,
        "x_hat": x_hat,
        "y_hat": y_hat,
        "h_hat": h_hat,
        "active_mask": active_mask,
        "solver": solver,
        "scales": scales,
        "n_p": n_p,
        "n_m": n_m,
    }

ForcingFunc = Callable[[float, float, Dict[str, float]], float]

def run_simulation(
    cfg: Config,
    initial_state: Optional[Dict[str, np.ndarray]] = None,
    forcing_func: Optional[ForcingFunc] = None,
    monitor_idx: Optional[int] = None,
    snapshot_tau: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    """
    Itera sobre o domínio temporal resolvendo o sistema linear em cada passo.
    
    O algoritmo implementa uma marcha no tempo.
    O vetor do lado direito (RHS) é atualizado em cada iteração contendo os termos de memória 
    (w e v do passo anterior) e os carregamentos externos (pressão no inlet).
    """
    system = prepare_system(cfg)
    n_m, n_p = int(system["n_m"]), int(system["n_p"])
    M, U = system["M"], system["U"]
    solver = system["solver"]
    h_hat = float(system["h_hat"])
    scales = system["scales"]

    # Criação do vetor de tempo. A adição de 0.5 * dt_hat lida com imprecisões de float no arange.
    tau = np.arange(0.0, cfg.tau_final + 0.5 * cfg.dt_hat, cfg.dt_hat)
    t_phys = tau * scales["t_ref"]
    w = np.zeros(n_m, dtype=float)
    v = np.zeros(n_m, dtype=float)
    
    if initial_state is not None:
        w[:] = initial_state.get("w", w)
        v[:] = initial_state.get("v", v)

    if monitor_idx is None:
        monitor_idx = ij2n(cfg.Nx // 2, cfg.Ny // 2, cfg.Nx)

    active_indices = U.getrow(cfg.outlet_node).indices
    vol = 0.0

    # Pré-alocação de memória para os vetores de histórico (melhor performance que o método .append() em listas)
    hist = {
        "tau": tau,
        "t_phys": t_phys,
        "p_inlet": np.zeros_like(tau),
        "p_outlet": np.zeros_like(tau),
        "q_outlet": np.zeros_like(tau),
        "w_center": np.zeros_like(tau),
        "w_monitor": np.zeros_like(tau),
        "volume": np.zeros_like(tau),
        "power": np.zeros_like(tau),
        "w_max_abs": np.zeros_like(tau),
    }
    
    snapshots: Dict[float, np.ndarray] = {}
    snapshot_tau = [] if snapshot_tau is None else list(snapshot_tau)
    snapshot_taken = {float(s): False for s in snapshot_tau}

    last_p = np.zeros(n_p, dtype=float)
    
    # Loop de Marcha no Tempo
    for k, tau_k in enumerate(tau):
        # Constrói vetor B do sistema A*x = B
        rhs = np.zeros(2 * n_m + n_p, dtype=float)
        rhs[:n_m] = w / cfg.dt_hat               # Inércia cinemática
        rhs[n_m:2 * n_m] = M.dot(v) / cfg.dt_hat # Momento linear anterior

        # Atualiza o forçamento (se houver, como p_inlet caindo a zero ou harmônico)
        p_in = forcing_func(float(tau_k), float(t_phys[k]), scales) if forcing_func else cfg.p_inlet
        rhs[2 * n_m + cfg.inlet_node] = p_in / scales["p_ref"] # Aplica BC de Dirichlet no nó da matriz penalizado

        # Solve via substituição retroativa (Forward/Backward substitution da LU)
        sol = solver.solve(rhs)
        
        # Desempacota o vetor solução (unpacking)
        w = sol[:n_m]
        v = sol[n_m:2 * n_m]
        p = sol[2 * n_m:]
        last_p = p.copy()

        # Pós-processamento e redimensionalização (volta ao domínio físico)
        p_out = p[cfg.outlet_node] * scales["p_ref"]
        
        # Integração numérica da vazão usando regra do retângulo: sum(v) * dA (sendo dA = h_hat^2 * R^2)
        q_out = (h_hat ** 2) * np.sum(v[active_indices]) * scales["v_ref"] * cfg.radius ** 2
        
        # Integração temporal explícita para o volume acumulado
        vol += q_out * cfg.dt_hat * scales["t_ref"]

        # Armazenamento de variáveis escalares (probe points)
        hist["p_inlet"][k] = p_in
        hist["p_outlet"][k] = p_out
        hist["q_outlet"][k] = q_out
        hist["w_center"][k] = w[ij2n(cfg.Nx // 2, cfg.Ny // 2, cfg.Nx)] * scales["w_ref"]
        hist["w_monitor"][k] = w[monitor_idx] * scales["w_ref"]
        hist["volume"][k] = vol
        hist["power"][k] = p_in * q_out  # Potência P = p * Q
        hist["w_max_abs"][k] = np.max(np.abs(w)) * scales["w_ref"]

        # Captura de campo completo (snapshots) em tempos específicos.
        # Risco: Salvar arrays de campo completo a cada passo de tempo esgotaria a RAM. 
        # A escolha de um array de tempos específicos contorna isso.
        for s in snapshot_tau:
            s = float(s)
            if not snapshot_taken[s] and tau_k >= s:
                snapshots[s] = w.copy() * scales["w_ref"]
                snapshot_taken[s] = True

    state = {"w": w.copy(), "v": v.copy(), "p": last_p.copy(), "snapshots": snapshots}
    return hist, state, system

# ============================================================
# PLOTAGEM COM ESCALAS LEGÍVEIS E EXPORTAÇÃO
# ============================================================

def ensure_output_dir(cfg: Config, sub: str = "") -> Path:
    """Garante a existência da estrutura de pastas para os resultados."""
    out = Path(cfg.output_dir)
    if sub:
        out = out / sub
    out.mkdir(parents=True, exist_ok=True)
    return out

def choose_scale(values: np.ndarray, kind: str) -> Tuple[float, str]:
    """
    Função de formatação de engenharia.
    Escala automaticamente os arrays para evitar eixos de gráficos com notação científica 
    excessiva (ex: 1e-6 m vira 1 µm). Isso é fundamental para a clareza da apresentação técnica.
    """
    maxabs = float(np.nanmax(np.abs(values))) if values.size else 0.0
    if maxabs == 0.0:
        return 1.0, ""
    if kind == "w":
        if maxabs < 1e-6: return 1e9, "nm"
        if maxabs < 1e-3: return 1e6, "µm"
        return 1e3, "mm"
    if kind == "p":
        if maxabs >= 1e3: return 1e-3, "kPa"
        return 1.0, "Pa"
    if kind == "q":
        if maxabs < 1e-12: return 1e15, "pL/s"
        if maxabs < 1e-9: return 1e12, "nL/s"
        if maxabs < 1e-6: return 1e9, "µL/s"
        return 1e6, "mL/s"
    if kind == "V":
        if maxabs < 1e-12: return 1e15, "pL"
        if maxabs < 1e-9: return 1e12, "nL"
        if maxabs < 1e-6: return 1e9, "µL"
        return 1e6, "mL"
    if kind == "P":
        if maxabs < 1e-9: return 1e12, "pW"
        if maxabs < 1e-6: return 1e9, "nW"
        if maxabs < 1e-3: return 1e6, "µW"
        if maxabs < 1.0: return 1e3, "mW"
        return 1.0, "W"
    return 1.0, ""

def plot_timeseries(tau: np.ndarray, data: np.ndarray, ylabel_base: str, kind: str, title: str, path: Path, label: Optional[str] = None) -> None:
    """Gera gráficos de linha simples para o histórico temporal."""
    scale, unit = choose_scale(np.asarray(data), kind)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(tau, data * scale, linewidth=1.8, label=label)
    ax.set_xlabel(r"Tempo adimensional $\hat{t}$")
    ax.set_ylabel(f"{ylabel_base} [{unit}]" if unit else ylabel_base)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.45)
    if label:
        ax.legend()
    ax.margins(x=0.02, y=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)

def plot_multi(tau: np.ndarray, curves: Sequence[Tuple[str, np.ndarray]], ylabel_base: str, kind: str, title: str, path: Path) -> None:
    """Gera gráficos com múltiplas curvas para comparação paramétrica."""
    all_vals = np.concatenate([np.asarray(c[1]) for c in curves])
    scale, unit = choose_scale(all_vals, kind)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for label, values in curves:
        ax.plot(tau, values * scale, linewidth=1.6, label=label)
    ax.set_xlabel(r"Tempo adimensional $\hat{t}$")
    ax.set_ylabel(f"{ylabel_base} [{unit}]" if unit else ylabel_base)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(fontsize=9)
    ax.margins(x=0.02, y=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)

def plot_deflection_profile(cfg: Config, system: Dict[str, object], snapshots: Dict[float, np.ndarray], path: Path) -> None:
    """
    Renderiza os campos 2D de deflexão (contour plots).
    Usa a máscara ativa (active_mask) para ocultar os nós penalizados fora do domínio circular,
    evitando distorções visuais nas bordas.
    """
    x = system["x_hat"]
    y = system["y_hat"]
    active_mask = system["active_mask"]
    Nx, Ny = cfg.Nx, cfg.Ny
    if not snapshots:
        return
    all_w = np.concatenate([np.asarray(w) for w in snapshots.values()])
    scale, unit = choose_scale(all_w, "w")
    vmax = np.nanmax(np.abs(all_w * scale))
    vmin = -vmax
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    X, Y = np.meshgrid(x, y)
    last_cf = None
    for ax, (tau_s, w_flat) in zip(axes[0], sorted(snapshots.items())):
        W = w_flat.reshape((Ny, Nx)) * scale
        W = np.ma.array(W, mask=~active_mask)
        last_cf = ax.contourf(X, Y, W, levels=24, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_title(rf"$\hat{{t}}={tau_s:g}$")
        ax.set_xlabel(r"$\hat{x}$")
        ax.set_ylabel(r"$\hat{y}$")
    cbar = fig.colorbar(last_cf, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label(f"Deflexão [{unit}]")
    fig.suptitle("Perfil transiente de deflexão da membrana", y=1.02)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)

def save_history_csv(hist: Dict[str, np.ndarray], path: Path) -> None:
    """Serializa os vetores de histórico temporal para CSV garantindo precisão numérica (12 casas)."""
    keys = ["tau", "t_phys", "p_inlet", "p_outlet", "q_outlet", "w_center", "w_monitor", "volume", "power", "w_max_abs"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for row in zip(*[hist[k] for k in keys]):
            writer.writerow([f"{float(x):.12e}" for x in row])

def summarize_case(hist: Dict[str, np.ndarray], cfg: Config, tag: str) -> Dict[str, float | str]:
    """Extrai escalares de interesse (picos e valores finais) de uma simulação para a tabela resumo."""
    return {
        "case": tag,
        "Nx": cfg.Nx,
        "Ny": cfg.Ny,
        "dt_hat": cfg.dt_hat,
        "tau_final": cfg.tau_final,
        "H_um": cfg.channel_width * 1e6,
        "p_inlet_Pa": cfg.p_inlet,
        "beta_hat": cfg.beta_hat,
        "p_out_final_Pa": float(hist["p_outlet"][-1]),
        "p_out_max_abs_Pa": float(np.max(np.abs(hist["p_outlet"]))),
        "q_out_max_abs_m3_s": float(np.max(np.abs(hist["q_outlet"]))),
        "w_center_final_m": float(hist["w_center"][-1]),
        "w_center_max_abs_m": float(np.max(np.abs(hist["w_center"]))),
        "volume_final_m3": float(hist["volume"][-1]),
        "power_max_abs_W": float(np.max(np.abs(hist["power"]))),
    }

# ============================================================
# ROTINAS DOS EXERCÍCIOS FINAIS
# ============================================================

def rotina_1_matriz_R(cfg: Config) -> None:
    """
    Tópico 1: Visualização da Matriz de Resistência Acoplada (R).
    
    Atenção à álgebra linear computacional aqui:
    A fórmula analítica é R = h_hat^2 * U^T * A_hat^-1 * U.
    Esta é essencialmente o Complemento de Schur do sistema hidráulico projetado na estrutura.
    
    Decisão de Projeto Crítica:
    NUNCA invertemos A explicitamente na programação numérica (inverter destrói a esparsidade e eleva o custo).
    Em vez de: A_inv = inv(A); R = U^T * A_inv * U
    Fazemos: Resolvemos o sistema (A * X = U) para obter X. Depois multiplicamos R = U^T * X.
    """
    out = ensure_output_dir(cfg, "topico_1")
    cfg1 = replace(cfg, Nx=26, Ny=26)
    system = prepare_system(cfg1)
    A = system["A_hat"]
    U = system["U"]
    h_hat = float(system["h_hat"])

    X = splinalg.spsolve(A.tocsc(), U.toarray())
    Rmat = h_hat ** 2 * (U.T @ X)
    R_dense = np.asarray(Rmat)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.spy(np.abs(R_dense) > 1e-14, marker=",", color="black")
    ax.set_title(r"Estrutura de esparsidade de $\mathbb{R}=\hat{h}^2\mathbb{U}^T\mathbb{A}^{-1}\mathbb{U}$")
    fig.tight_layout()
    fig.savefig(out / "topico1_spy_matriz_R.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    with open(out / "topico1_resumo.txt", "w", encoding="utf-8") as f:
        f.write("Tópico 1 - Matriz R equivalente\n")
        f.write(f"Dimensão de R: {R_dense.shape[0]} x {R_dense.shape[1]}\n")
        f.write(f"nnz visualizados: {int(np.count_nonzero(np.abs(R_dense) > 1e-14))}\n")
        f.write("R = h_hat^2 U^T A_hat^{-1} U, após imposição de p_inlet no nó 0.\n")

def rotina_2_evolucao_parametrica(cfg: Config, full_sweep: bool = False) -> None:
    """
    Tópico 2: Avaliação Transiente e Varredura Paramétrica.
    
    Utiliza replace() do dataclass para criar configurações mutadas derivadas da base,
    garantindo que não haja poluição de estado entre as execuções do loop (thread-safe na essência).
    """
    out = ensure_output_dir(cfg, "topico_2")
    summary_rows: List[Dict[str, float | str]] = []

    # Caso-base para os gráficos principais
    base = replace(cfg, Nx=51, Ny=51, dt_hat=0.025, tau_final=12.0,
                   p_inlet=5000.0, channel_width=1000e-6, beta_hat=0.1)
    hist, state, system = run_simulation(base, snapshot_tau=[0, 1, 3, 6, 12])
    save_history_csv(hist, out / "topico2_caso_base_historico.csv")
    summary_rows.append(summarize_case(hist, base, "base_N51_dt0025_p5000_H1000"))

    plot_deflection_profile(base, system, state["snapshots"], out / "topico2_perfil_transiente_deflexao.png")
    # (... geração das imagens das séries temporais omitida nos comentários por ser autoexplicativa ...)
    plot_timeseries(hist["tau"], hist["p_outlet"], r"$p_{outlet}$", "p", "Tópico 2 - Pressão no nó de descarga", out / "topico2_pressao_outlet.png")
    plot_timeseries(hist["tau"], hist["q_outlet"], r"$q_{outlet}$", "q", "Tópico 2 - Vazão de saída", out / "topico2_vazao_outlet.png")
    plot_timeseries(hist["tau"], hist["w_center"], "Deslocamento central", "w", "Tópico 2 - Deslocamento vertical central", out / "topico2_deslocamento_central.png")
    plot_timeseries(hist["tau"], hist["w_max_abs"], "Deflexão máxima absoluta", "w", "Tópico 2 - Perfil transiente resumido por deflexão máxima", out / "topico2_deflexao_maxima.png")
    plot_timeseries(hist["tau"], hist["volume"], "Volume acumulado", "V", "Tópico 2 - Volume acumulado no reservatório", out / "topico2_volume_acumulado.png")
    plot_timeseries(hist["tau"], hist["power"], "Potência de entrada", "P", "Tópico 2 - Potência consumida", out / "topico2_potencia_consumida.png")

    # Varreduras e comparações (Pressão, H, dt, Malha)
    curves_p = []
    for p_in in [5e3, 1e4, 2e4]:
        c = replace(base, p_inlet=float(p_in))
        h, _, _ = run_simulation(c)
        summary_rows.append(summarize_case(h, c, f"pressao_p{int(p_in)}"))
        curves_p.append((f"p_in={p_in/1000:g} kPa", h["w_center"]))
    plot_multi(hist["tau"], curves_p, "Deslocamento central", "w", "Influência da pressão de entrada", out / "topico2_comparacao_pressoes.png")

    curves_H = []
    for H_um in [1000, 1250, 1500, 1750]:
        c = replace(base, channel_width=H_um * 1e-6)
        h, _, _ = run_simulation(c)
        summary_rows.append(summarize_case(h, c, f"largura_H{H_um}"))
        curves_H.append((f"H={H_um} µm", h["p_outlet"]))
    plot_multi(hist["tau"], curves_H, r"$p_{outlet}$", "p", "Influência da largura dos canais", out / "topico2_comparacao_larguras_pressao.png")

    curves_dt = []
    for dt_hat in [0.00625, 0.0125, 0.025, 0.05]:
        c = replace(base, dt_hat=dt_hat)
        h, _, _ = run_simulation(c)
        summary_rows.append(summarize_case(h, c, f"dt_{dt_hat:g}"))
        curves_dt.append((rf"$\Delta\hat{{t}}$={dt_hat:g}", h["w_center"]))
    
    # Tratamento customizado para passos de tempo variados (tamanhos de vetores tau diferentes)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    all_vals = np.concatenate([c[1] for c in curves_dt])
    scale, unit = choose_scale(all_vals, "w")
    for dt_hat, (label, values) in zip([0.00625, 0.0125, 0.025, 0.05], curves_dt):
        tau_dt = np.arange(0.0, 12.0 + 0.5 * dt_hat, dt_hat)
        ax.plot(tau_dt, values * scale, linewidth=1.3, label=label)
    ax.set_xlabel(r"Tempo adimensional $\hat{t}$")
    ax.set_ylabel(f"Deslocamento central [{unit}]")
    ax.set_title("Comparação entre passos de tempo adimensionais")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(fontsize=9)
    ax.margins(x=0.02, y=0.15)
    fig.tight_layout()
    fig.savefig(out / "topico2_comparacao_dt.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    curves_mesh = []
    for N in [51, 101]:
        c = replace(base, Nx=N, Ny=N)
        h, _, _ = run_simulation(c)
        summary_rows.append(summarize_case(h, c, f"malha_{N}"))
        curves_mesh.append((f"{N}x{N}", h["w_center"]))
    plot_multi(hist["tau"], curves_mesh, "Deslocamento central", "w", "Comparação entre malhas 51x51 e 101x101", out / "topico2_comparacao_malhas.png")

    if full_sweep:
        sweep_out = out / "varredura_completa"
        sweep_out.mkdir(exist_ok=True)
        for N in [51, 101]:
            for dt_hat in [0.00625, 0.0125, 0.025, 0.05]:
                for p_in in [5e3, 1e4, 2e4]:
                    for H_um in [1000, 1250, 1500, 1750]:
                        c = replace(cfg, Nx=N, Ny=N, dt_hat=dt_hat, tau_final=12.0,
                                    p_inlet=float(p_in), channel_width=H_um * 1e-6, beta_hat=0.1)
                        h, _, _ = run_simulation(c)
                        tag = f"N{N}_dt{dt_hat:g}_p{int(p_in)}_H{H_um}"
                        summary_rows.append(summarize_case(h, c, tag))
                        save_history_csv(h, sweep_out / f"{tag}.csv")

    pd.DataFrame(summary_rows).to_csv(out / "topico2_resumo_parametrico.csv", index=False)

def rotina_3_queda_pressao(cfg: Config) -> None:
    """
    Tópico 3: Problema de Relaxamento.
    Usa o estado final (w, v) do Tópico 2 como condição inicial, 
    zerando o forçamento externo e observando o escoamento provocado pelo retorno elástico da membrana.
    """
    out = ensure_output_dir(cfg, "topico_3")
    base = replace(cfg, Nx=51, Ny=51, dt_hat=0.025, tau_final=12.0,
                   p_inlet=5000.0, channel_width=1000e-6, beta_hat=0.1)
    _, state_final, _ = run_simulation(base)

    queda = replace(base, p_inlet=0.0)
    hist, _, _ = run_simulation(queda, initial_state={"w": state_final["w"], "v": state_final["v"]})
    save_history_csv(hist, out / "topico3_queda_pressao_historico.csv")
    plot_timeseries(hist["tau"], hist["p_outlet"], r"$p_{outlet}$", "p", "Tópico 3 - Relaxamento após queda de pressão", out / "topico3_pressao_outlet.png")
    plot_timeseries(hist["tau"], hist["q_outlet"], r"$q_{outlet}$", "q", "Tópico 3 - Vazão durante relaxamento", out / "topico3_vazao_outlet.png")
    plot_timeseries(hist["tau"], hist["w_center"], "Deslocamento central", "w", "Tópico 3 - Deflexão central após p_inlet=0", out / "topico3_deflexao_central.png")
    plot_timeseries(hist["tau"], hist["volume"], "Volume acumulado", "V", "Tópico 3 - Volume após queda de pressão", out / "topico3_volume.png")
    plot_timeseries(hist["tau"], hist["power"], "Potência de entrada", "P", "Tópico 3 - Potência após queda de pressão", out / "topico3_potencia.png")
    pd.DataFrame([summarize_case(hist, queda, "queda_pressao")]).to_csv(out / "topico3_resumo.csv", index=False)

def mode3_initial_state(cfg: Config) -> Tuple[Dict[str, np.ndarray], int, float, float, Dict[str, object]]:
    """Configura a condição inicial baseada na forma modal (3º Modo)."""
    system = prepare_system(cfg)
    evals, evecs, omega = solve_membrane_modes(system["K"], system["M"], num_modes=8)
    mode_index = 2 # Índice 2 representa o 3º modo (0, 1, 2)
    phi = evecs[:, mode_index].copy()
    phi /= np.max(np.abs(phi))
    phi *= cfg.modal_initial_amplitude_hat
    
    # Fundamental: monitorar o deslocamento no antinó (pico) da forma de onda. 
    # Se monitorássemos o centro em modos assimétricos (ex: onde w=0), não veríamos a oscilação.
    monitor_idx = int(np.argmax(np.abs(phi)))
    omega3_hat = float(omega[mode_index])
    f3_phys = omega3_hat / (2.0 * np.pi * system["scales"]["t_ref"])
    return {"w": phi, "v": np.zeros(cfg.Nx * cfg.Ny)}, monitor_idx, omega3_hat, f3_phys, system

def estimate_frequency_from_zero_crossings(tau: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Estima a frequência natural numérica observando os cruzamentos por zero do histórico.
    
    Técnica: Interpolação Linear de Submalha.
    Procurar onde s_i * s_{i+1} < 0 indica que cruzou o zero, mas o cruzamento exato 
    está entre os passos temporais dt. Interpolar: t_zero = t0 - y0 * (t1 - t0) / (y1 - y0)
    aumenta a precisão da estimativa de frequência além do limite imposto pelo dt da simulação.
    """
    y = np.asarray(y)
    # Remove a média final para centralizar a oscilação amortecida
    y = y - np.mean(y[-max(3, len(y)//10):])
    s = np.sign(y)
    s[s == 0] = 1
    idx = np.where(np.diff(s) != 0)[0]
    if len(idx) < 3:
        return float("nan"), float("nan")
    crossings = []
    for i in idx:
        t0, t1 = tau[i], tau[i + 1]
        y0, y1 = y[i], y[i + 1]
        if y1 == y0:
            crossings.append(t0)
        else:
            crossings.append(t0 - y0 * (t1 - t0) / (y1 - y0))
    crossings = np.asarray(crossings)
    half_periods = np.diff(crossings)
    period_hat = 2.0 * float(np.mean(half_periods))
    omega_hat = 2.0 * np.pi / period_hat
    return omega_hat, period_hat

def rotina_4_oscilacao_livre(cfg: Config) -> None:
    """Tópico 4: Dinâmica Livre. Compara a frequência natural puramente estrutural com a simulada."""
    out = ensure_output_dir(cfg, "topico_4")
    c = replace(cfg, Nx=51, Ny=51, channel_width=2000e-6, beta_hat=0.0, p_inlet=0.0,
                dt_hat=0.00625, tau_final=40.0)
    initial, monitor_idx, omega3_hat, f3_phys, _ = mode3_initial_state(c)
    hist, _, system = run_simulation(c, initial_state=initial, monitor_idx=monitor_idx)
    omega_sim_hat, period_hat = estimate_frequency_from_zero_crossings(hist["tau"], hist["w_monitor"])
    f_sim_phys = omega_sim_hat / (2.0 * np.pi * system["scales"]["t_ref"]) if np.isfinite(omega_sim_hat) else float("nan")

    save_history_csv(hist, out / "topico4_oscilacao_livre_historico.csv")
    plot_timeseries(hist["tau"], hist["w_monitor"], "Deslocamento no ponto modal", "w", "Tópico 4 - Oscilação livre a partir do 3º modo", out / "topico4_deslocamento_modal.png")
    plot_timeseries(hist["tau"], hist["p_outlet"], r"$p_{outlet}$", "p", "Tópico 4 - Pressão induzida na oscilação livre", out / "topico4_pressao_outlet.png")

    with open(out / "topico4_frequencias.txt", "w", encoding="utf-8") as f:
        f.write("Tópico 4 - Frequência do terceiro modo\n")
        f.write(f"omega3_hat isolada = {omega3_hat:.8e}\n")
        f.write(f"freq3 física isolada = {f3_phys:.8e} Hz\n")
        f.write(f"omega_hat simulada = {omega_sim_hat:.8e}\n")
        f.write(f"freq física simulada = {f_sim_phys:.8e} Hz\n")
        f.write(f"período adimensional estimado = {period_hat:.8e}\n")
        f.write(f"monitor_idx = {monitor_idx}\n")

def rotina_5_forcamento_harmonico(cfg: Config) -> None:
    """
    Tópico 5: Forçamento Harmônico na Frequência Natural (Ressonância).
    Submete o sistema a um termo fonte p_inlet dependente do tempo.
    """
    out = ensure_output_dir(cfg, "topico_5")
    c_modes = replace(cfg, Nx=51, Ny=51, channel_width=2000e-6, beta_hat=0.0, p_inlet=0.0,
                      dt_hat=0.00625, tau_final=1.0)
    _, monitor_idx, omega3_hat, f3_phys, _ = mode3_initial_state(c_modes)

    c = replace(c_modes, tau_final=40.0)

    def forcing(tau_hat: float, t_phys: float, scales: Dict[str, float]) -> float:
        # A coerência aqui é dada por forçar a rede hidráulica com a mesma pulsação natural do modo estrutural
        return 5000.0 * math.cos(omega3_hat * tau_hat)

    hist, _, _ = run_simulation(c, forcing_func=forcing, monitor_idx=monitor_idx)
    save_history_csv(hist, out / "topico5_forcamento_harmonico_historico.csv")
    plot_timeseries(hist["tau"], hist["p_inlet"], r"$p_{inlet}$", "p", "Tópico 5 - Entrada harmônica", out / "topico5_pressao_entrada.png")
    plot_timeseries(hist["tau"], hist["w_monitor"], "Deslocamento no ponto modal", "w", "Tópico 5 - Resposta ao forçamento harmônico", out / "topico5_deslocamento_modal.png")
    plot_timeseries(hist["tau"], hist["p_outlet"], r"$p_{outlet}$", "p", "Tópico 5 - Pressão no outlet sob forçamento harmônico", out / "topico5_pressao_outlet.png")
    plot_timeseries(hist["tau"], hist["power"], "Potência de entrada", "P", "Tópico 5 - Potência sob forçamento harmônico", out / "topico5_potencia.png")

    with open(out / "topico5_parametros.txt", "w", encoding="utf-8") as f:
        f.write("Tópico 5 - Forçamento harmônico\n")
        f.write("p_inlet(tau) = 5000*cos(omega3_hat*tau) Pa\n")
        f.write(f"omega3_hat = {omega3_hat:.8e}\n")
        f.write(f"freq3 física equivalente = {f3_phys:.8e} Hz\n")
        f.write(f"monitor_idx = {monitor_idx}\n")

def write_global_readme(cfg: Config, topics: Sequence[int], full_sweep: bool) -> None:
    """Gera metadados informativos na pasta de saída."""
    out = ensure_output_dir(cfg)
    scales = reference_scales(cfg)
    with open(out / "LEIA_ME_resultados.txt", "w", encoding="utf-8") as f:
        f.write("Resultados do acoplamento hidráulico-mecânico\n")
        f.write("=====================================================\n\n")
        f.write("Fundamento: exercícios finais da seção 5.2.5 do PDF.\n")
        f.write("O eixo temporal dos gráficos é o tempo adimensional tau=t/t_ref.\n\n")
        f.write("Escalas de referência:\n")
        for k, v in scales.items():
            f.write(f"  {k}: {v:.12e}\n")
        f.write("\nTópicos executados: " + ", ".join(map(str, topics)) + "\n")
        f.write(f"Varredura completa do item 2: {full_sweep}\n")
        f.write("\nObservação: network_length_unit=1e-3 m foi usado para converter as coordenadas do grafo em comprimentos físicos.\n")

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def parse_args() -> argparse.Namespace:
    """Gestão de argumentos da interface de linha de comando."""
    parser = argparse.ArgumentParser(description="Rotinas corrigidas para o acoplamento hidráulico-mecânico do PDF.")
    parser.add_argument("--topics", type=int, nargs="*", default=[1, 2, 3, 4, 5], help="Tópicos a executar: 1 2 3 4 5")
    parser.add_argument("--quick", action="store_true", help="Executa uma versão rápida e apresentável. Não roda a varredura completa de 96 casos.")
    parser.add_argument("--full-sweep", action="store_true", help="No tópico 2, salva CSV para todos os 96 casos pedidos no PDF.")
    parser.add_argument("--output", type=str, default="resultados_acoplamento", help="Pasta de saída")
    return parser.parse_args()

def main() -> None:
    """Ponto de entrada (Entry point) do script."""
    args = parse_args()
    cfg = replace(Config(), output_dir=args.output)
    topics = args.topics
    t0 = time.time()
    
    write_global_readme(cfg, topics, args.full_sweep)
    
    # Controle de fluxo via argumentos de linha de comando
    if 1 in topics:
        rotina_1_matriz_R(cfg)
    if 2 in topics:
        rotina_2_evolucao_parametrica(cfg, full_sweep=args.full_sweep)
    if 3 in topics:
        rotina_3_queda_pressao(cfg)
    if 4 in topics:
        rotina_4_oscilacao_livre(cfg)
    if 5 in topics:
        rotina_5_forcamento_harmonico(cfg)
        
    print(f"Concluído. Resultados em: {Path(cfg.output_dir).resolve()}")
    print(f"Tempo total: {time.time() - t0:.2f} s")

# Protege a execução para que o main() só rode se o script for chamado diretamente,
# mantendo o arquivo seguro caso queira ser importado como biblioteca em outro lugar.
if __name__ == "__main__":
    main()