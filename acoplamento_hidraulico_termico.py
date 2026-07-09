# -*- coding: utf-8 -*-
"""
acoplamento_hidraulico_termico_integrado.py
============================================================
INTEGRAÇÃO HIDRÁULICO-TÉRMICA - CAPÍTULO 4
============================================================

INTUIÇÃO TÉCNICA DO ACOPLAMENTO:
O problema trata de dois domínios físicos que interagem:
1. Placa Térmica (Condução 2D): Descrita por uma Equação Diferencial Parcial (EDP) de Poisson
   com difusão k(x,y) e termo fonte S(x,y). Discretizada via malha regular.
2. Rede Hidráulica (Escoamento 1D em rede): Descrita pela conservação de massa em grafos, 
   onde o fluxo depende da diferença de pressão e da condutância.

O acoplamento ("one-way" ou iterativo) ocorre porque a temperatura afeta a viscosidade 
do fluido (μ), alterando a condutância dos canais. Inversamente, a presença dos canais 
atua como sorvedouro de calor na placa.

Este arquivo integra a parte térmica e a parte hidráulica preservando os
nomes centrais já usados nos códigos originais:

PARTE TÉRMICA PRESERVADA
- ThermalPlateConfig
- BaseLinearSystemSolver
- ThermalPlateSolver
- ThermalPlateSolver.assembly
- ThermalPlateSolver.solve_system
- ij2n, n2ij
- constant_source, top_bottom_temperature_function

PARTE HIDRÁULICA PRESERVADA
- water_viscosity_pa_s
- empirical_viscosity
- edge_lengths
- build_incidence_matrix
- assembly                  -> montagem global hidráulica
- evaluate_flow_bc
- apply_pressure_bc
- hydraulic_conductivities
- solve_network
- compute_power
- nodal_mass_residual
- print_inputs_summary
- print_output_summary
- print_final_explanation

O QUE FOI ACRESCENTADO PARA O CAPÍTULO 4
- TemperatureInterpolator: interpolação 2D linear, cubic e nearest.
- Regras de quadratura em arestas: ponto médio e trapézio, simples e compostas.
- HydroThermalCoupledModel: integra placa térmica + rede hidráulica.
- NetworkInfluenceModel: efeito dos microcanais na condutividade térmica e no termo fonte/sumidouro.
- Funções de exercícios do PDF, focadas nas perguntas destacadas em amarelo:
  4.2.1 itens 1 a 5 e 4.3.3 itens 1 e 2.

DEPENDÊNCIAS
    pip install numpy scipy matplotlib pandas tabulate shapely

SAÍDAS GERADAS
- Pasta: resultados_acoplamento/
- Figuras PNG com mapas de temperatura, rede colorida por nós/arestas e perfis.
- CSVs com tabelas de comparação numérica.

OBSERVAÇÃO IMPORTANTE
O docente disponibiliza uma função externa generate_graph_arrays para montar a rede.
Se os arquivos gera_grafo.py e plota_rede.py estiverem na mesma pasta, o código usa a
função original. Caso contrário, há um gerador substituto compatível, para que o arquivo
rode de forma autônoma e ainda mantenha os nós 0 e 175 exigidos no enunciado.
"""

# Permite o uso de anotações de tipo de classes que ainda não foram definidas no escopo atual.
# Essencial para type hinting limpo e legível (ex: Type hinting de métodos que retornam instâncias da própria classe).
from __future__ import annotations

# ============================================================
# IMPORTS E FUNDAMENTAÇÃO DAS BIBLIOTECAS
# ============================================================

# Manipulação de caminhos de arquivo orientada a objetos.
# Trade-off: Ligeiramente mais verboso que os.path, mas garante compatibilidade cross-platform (Windows usa '\', Unix usa '/') sem risco de bugs de concatenação de strings.
from pathlib import Path

# Biblioteca padrão para medição de tempo de CPU.
# Utilizada aqui para benchmarking (profiling) das funções de montagem e solução dos sistemas.
import time

# Fornece decoradores para criar classes de configuração.
# Vantagem: Reduz o boilerplate de métodos __init__ e facilita a imutabilidade/organização dos parâmetros físicos do problema.
from dataclasses import dataclass, field

# Sistema de tipagem estática (Type Hinting).
# Ajuda analisadores estáticos (como mypy) a encontrar bugs antes da execução e serve como documentação viva do contrato das funções.
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# NumPy: Núcleo matemático do script.
# Fundamento: O Python puro é lento para iterações extensas. O NumPy delega as operações matriciais e vetoriais para C/Fortran, garantindo eficiência.
import numpy as np

# Pandas: Utilizado estritamente para estruturação tabular de dados e exportação de relatórios (CSV/Markdown).
# Limitação: Para cálculos numéricos diretos de EDPs, o Pandas introduz um overhead desnecessário; por isso, é restrito apenas à camada de saída de dados.
import pandas as pd

# Matplotlib: Motor de plotagem.
import matplotlib
# Configuração do backend 'Agg' (Anti-Grain Geometry).
# Fundamento: Como o script pode ser rodado em servidores ou terminais sem interface gráfica (X11/Wayland), 
# o backend Agg renderiza as imagens diretamente em arquivos (PNG), evitando travamentos (segmentation faults) relacionados a janelas de exibição.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# SciPy - Matrizes Esparsas e Álgebra Linear.
# Fundamento: Em Malhas de Volumes Finitos ou Diferenças Finitas 2D, a matriz do sistema é gigantesca, mas a maioria dos elementos é zero (cada nó só interage com seus 4 vizinhos).
# Trade-off: Armazenar isso como matriz densa (NumPy tradicional) tem complexidade de memória O(N^2). Matrizes esparsas (CSR/COO) reduzem para O(N), viabilizando malhas refinadas.
from scipy import sparse
from scipy.sparse.linalg import spsolve

# SciPy - Interpolação Bidimensional.
# Fundamento: A rede hidráulica possui nós em coordenadas arbitrárias, enquanto o modelo térmico resolve a temperatura em uma grade fixa. 
# RegularGridInterpolator (linear/nearest) e RectBivariateSpline (cúbico) permitem "mapear" dados da grade estruturada para os pontos do grafo.
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

# SciPy - Estruturas de Partição Espacial.
# Fundamento: cKDTree implementa uma árvore k-dimensional em C.
# Aplicação: Para encontrar qual nó ou aresta está próximo de um ponto na malha, uma busca por força bruta custaria O(N*M). A KDTree reduz o custo computacional médio das buscas espaciais para O(M log N).
from scipy.spatial import cKDTree

# ============================================================
# CONFIGURAÇÃO GERAL DO ESTUDO DO PDF
# ============================================================

# Diretório de saída criado de forma idempotente (exist_ok=True impede erro se a pasta já existir).
# Decisão de projeto: Isolar os resultados gerados mantém o diretório de trabalho limpo.
OUTPUT_DIR = Path("./resultados_acoplamento")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Centralização de "Magic Numbers". 
# Fundamento: Em simulações numéricas, espalhar constantes pelo código gera bugs difíceis de rastrear.
# Um dicionário global garante que qualquer alteração de malha ou geometria propague para todas as funções.
# Nota: Todas as dimensões físicas estão rigorosamente convertidas para o Sistema Internacional (SI) - Metros, Watts, Kelvin/Celsius, etc.
CONFIG_INTEGRADO: Dict[str, Any] = {
    # Geometria da placa convertida de cm para m.
    "Lx": 0.03,
    "Ly": 0.015,

    # Propriedades térmicas da placa.
    "k_constant": 0.25,
    "source_value": 5.0e5,
    "TL": 10.0,
    "TR": 30.0,
    "TC": 35.0,
    "circle_radius": 0.0025,

    # O centro do círculo é (2 + R, 0.75) cm = (0.02 + R, 0.0075) m.
    "circle_center_x": 0.02 + 0.0025,
    "circle_center_y": 0.0075,

    # Rede hidráulica.
    "levels": 3,
    "spine_length": 6,
    "coord_scale_to_m": 0.001,  # Fator de escala para converter o grafo (provavelmente em mm) para m.
    "width": 500e-6,            # 500 micrometros
    "height": 500e-6,           # 500 micrometros
    "area_constant": 500e-6 * 500e-6,
    "inlet_0": 0,
    "inlet_175": 175,
    "Q0_in": 1.0e-7,            # Vazão volumétrica no nó 0
    "Q175_in": 1.0e-6,          # Vazão volumétrica no nó 175

    # Malhas citadas nos exercícios do PDF.
    "thermal_mesh_refined": (241, 121),
    "thermal_mesh_coarse": (61, 31),
    "secondary_grid_refined_case": (61, 31),
    "secondary_grid_coarse_case": (31, 16),

    # Flags lógicas (booleanas) para controlar o fluxo de execução (feature toggles).
    # Trade-off: Permite rodar apenas uma parte do script para debugar, economizando tempo computacional.
    "run_ex_4-2-1_item-1": True,      # Deduzir regra do trapézio
    "run_ex_4-2-1_item_2": True,     # Interpolação e malhas
    "run_ex_4-2-1_item_3-4-5": True, # Quadratura e hidráulica
    "run_ex_4-3-3_item-1": True,     # Condutividade modificada
    "run_ex_4-3-3_item-2": True,     # Fonte/Sumidouro

    # Modo de execução do item 4.3.
    # Limitação prática: A malha (241,121) em loop no Python puro pode exigir muita RAM e tempo.
    # O default do código permite passar por malhas menores antes para garantir a execução.
    "thermal_meshes_43": [(61, 31), (121, 61), (241,121)],

    "dmax_values": [0.00025, 0.0005, 0.001],
    "s0_values": [1e5, -1e5, 5e5, -5e5, 1e6, -1e6],
}


# ============================================================
# TIPOS AUXILIARES DA PARTE TÉRMICA
# ============================================================

# Definição de alias de tipos (Type Aliases) para clareza semântica.
Number = Union[int, float]
# Aceita um número fixo ou uma função matemática que depende do espaço.
ScalarOrCallable = Union[Number, Callable[[float], float]]
# Representa um campo 2D contínuo, ex: f(x,y).
FieldFunction2D = Callable[[float, float], float]


# ============================================================
# FUNÇÕES DE MAPEAMENTO DE ÍNDICES DA PLACA
# ============================================================

def ij2n(i: int, j: int, Nx: int) -> int:
    """
    Converte o índice 2D (i,j) da malha em índice global n (Ordem Lexicográfica / Row-major).

    Intuição:
    Para resolver a EDP numericamente, a matriz da malha 2D precisa ser "achatada" (flatten) 
    em um vetor de incógnitas 1D (T). O sistema resultante é A * T = b.
    
    A fórmula n = i + j*Nx percorre primeiro os elementos na direção x (variação de i) 
    e depois "pula" para a próxima linha em y (variação de j).
    """
    return i + j * Nx


def n2ij(n: int, Nx: int) -> Tuple[int, int]:
    """
    Faz o caminho inverso: índice global n -> índices locais (i,j).
    Fundamento: A divisão inteira (//) recupera a linha (j), e o resto (%) recupera a coluna (i).
    """
    j = n // Nx
    i = n % Nx
    return i, j


# ============================================================
# FUNÇÕES DE CAMPO, FONTE E CONTORNO TÉRMICO
# ============================================================
# Paradigma Funcional: Utilização de Higher-Order Functions (funções que retornam funções).
# Trade-off: Adiciona um pequeno overhead de chamada de função no loop de montagem da matriz, 
# mas torna o solver universal. O solver não precisa saber se o contorno é uma constante 
# ou um polinômio; ele apenas avalia f(x) ou f(x,y).

def to_callable(value: ScalarOrCallable) -> Callable[[float], float]:
    """Transforma número constante em função para padronizar a interface de contornos."""
    if callable(value):
        return value
    # Se for escalar, retorna uma função lambda que ignora a entrada e devolve o escalar.
    return lambda _: float(value)


def constant_source(value: float) -> FieldFunction2D:
    """Retorna uma fonte volumétrica constante S(x,y) = value."""
    return lambda x, y: float(value)


def top_bottom_temperature_function(Lx: float) -> Callable[[float], float]:
    """
    Temperatura prescrita no topo e na base segundo o PDF.
    Condição de contorno de Dirichlet variável no espaço.
    """
    return lambda x: 10.0 + 20.0 * x / Lx


def variable_k_function(Lx: float, Ly: float) -> FieldFunction2D:
    """
    Condutividade térmica k(x,y) variável com comportamento senoidal.
    Mantida por retrocompatibilidade com o modelo térmico não-acoplado.
    """
    return lambda x, y: 0.2 + 0.05 * np.sin(3.0 * np.pi * x / Lx) * np.sin(3.0 * np.pi * y / Ly)

# ============================================================
# CONFIGURAÇÃO E SOLVER DA PLACA TÉRMICA
# ============================================================

@dataclass
class ThermalPlateConfig:
    """
    Configuração do problema térmico estacionário.

    Fundamento Matemático:
    Define o domínio e os parâmetros da Equação Diferencial Parcial (EDP) elíptica:
        -div(k grad T) = S
    com condições de Dirichlet (T prescrito) nas bordas e em uma região circular interna.

    Decisão de Projeto (@dataclass):
    Agrupa todos os hiperparâmetros. O uso do `field(default_factory=...)` evita 
    o anti-pattern de usar objetos mutáveis (como funções lambda ou listas) como 
    valores padrão diretamente na assinatura da classe.
    """

    Lx: float = 0.03
    Ly: float = 0.015
    Nx: int = 61
    Ny: int = 31

    # Condições de contorno e fonte. Aceitam escalares ou funções.
    TL: ScalarOrCallable = 10.0
    TR: ScalarOrCallable = 30.0
    TB: ScalarOrCallable = field(default_factory=lambda: top_bottom_temperature_function(0.03))
    TT: ScalarOrCallable = field(default_factory=lambda: top_bottom_temperature_function(0.03))

    source_function: FieldFunction2D = field(default_factory=lambda: constant_source(5.0e5))

    use_variable_k: bool = False
    k_constant: float = 0.25
    k_function: Optional[FieldFunction2D] = None

    use_circle_constraint: bool = True
    circle_center_x: float = 0.0225
    circle_center_y: float = 0.0075
    circle_radius: float = 0.0025
    TC: float = 35.0

    solver_mode: str = "sparse"
    # preserve_symmetry é fundamental se fôssemos usar métodos iterativos (como Gradientes Conjugados).
    # Matrizes simétricas e definidas positivas (SPD) garantem convergência mais rápida e estável.
    preserve_symmetry: bool = True

    output_dir: Path = OUTPUT_DIR
    contour_levels: int = 24

    def __post_init__(self) -> None:
        """
        Garante que os parâmetros fornecidos como números sejam promovidos a funções (Callables).
        Isso elimina verificações `if callable(x)` no loop principal do solver.
        """
        self.TL = to_callable(self.TL)
        self.TR = to_callable(self.TR)
        self.TB = to_callable(self.TB)
        self.TT = to_callable(self.TT)

        if self.use_variable_k and self.k_function is None:
            self.k_function = variable_k_function(self.Lx, self.Ly)


class BaseLinearSystemSolver:
    """
    Classe base para concentrar a lógica algébrica da resolução de A * T = b.
    
    Trade-off de Desempenho:
    Nunca se deve usar `np.linalg.inv(A) @ b` para resolver sistemas lineares grandes. 
    A inversão explícita é numericamente instável e custosa (O(N^3)). 
    O correto é fatorar o sistema (ex: LU ou Cholesky), que é o que `solve` e `spsolve` fazem por baixo dos panos.
    """

    def __init__(self) -> None:
        self.last_result: Optional[Dict[str, Any]] = None

    def solve_linear_system_dense(self, A_mod: np.ndarray, b_mod: np.ndarray) -> np.ndarray:
        return np.linalg.solve(A_mod, b_mod)

    def solve_linear_system_sparse(self, A_mod_sparse: sparse.csr_matrix, b_mod: np.ndarray) -> np.ndarray:
        return spsolve(A_mod_sparse, b_mod)


class ThermalPlateSolver(BaseLinearSystemSolver):
    """
    Solver térmico baseado no Método de Volumes Finitos (FVM) / Diferenças Finitas em malha estruturada.
    """

    def __init__(self, cfg: ThermalPlateConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Geração da malha nodal (coordenadas dos centros dos volumes de controle)
        self.x = np.linspace(0.0, cfg.Lx, cfg.Nx)
        self.y = np.linspace(0.0, cfg.Ly, cfg.Ny)

        # Resolução espacial. 
        # Limitação: Malhas uniformes (passo hx e hy constantes) são fáceis de implementar,
        # mas ineficientes se houverem altos gradientes locais (ex: ao redor do círculo ou canais).
        # Uma malha adaptativa seria um aprimoramento físico considerável.
        self.hx = cfg.Lx / (cfg.Nx - 1)
        self.hy = cfg.Ly / (cfg.Ny - 1)

        # Graus de liberdade (Número de incógnitas)
        self.nunk = cfg.Nx * cfg.Ny

        # Pré-processamento das condições de contorno
        self.dirichlet_ids, self.dirichlet_values = self._build_dirichlet_data()
        self.dirichlet_set = set(self.dirichlet_ids.tolist())
        self.dirichlet_value_by_id = {int(i): float(v) for i, v in zip(self.dirichlet_ids, self.dirichlet_values)}

    def _build_dirichlet_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Varre o domínio buscando nós que possuem temperatura imposta (Dirichlet)."""
        ids: List[int] = []
        vals: List[float] = []
        for j in range(self.cfg.Ny):
            for i in range(self.cfg.Nx):
                value = self.get_dirichlet_value_raw(i, j)
                if value is not None:
                    ids.append(ij2n(i, j, self.cfg.Nx))
                    vals.append(float(value))
        return np.array(ids, dtype=int), np.array(vals, dtype=float)

    def get_k(self, x: float, y: float) -> float:
        """Avalia a condutividade térmica no ponto."""
        if self.cfg.use_variable_k:
            if self.cfg.k_function is None:
                raise ValueError("use_variable_k=True, mas k_function não foi definida.")
            return float(self.cfg.k_function(x, y))
        return float(self.cfg.k_constant)

    def get_source(self, x: float, y: float) -> float:
        """Avalia o termo fonte volumétrico S(x,y)."""
        return float(self.cfg.source_function(x, y))

    def is_inside_circle_constraint(self, x: float, y: float) -> bool:
        """Testa se (x,y) cai dentro do raio do restritor circular."""
        if not self.cfg.use_circle_constraint:
            return False
        dx = x - self.cfg.circle_center_x
        dy = y - self.cfg.circle_center_y
        return dx * dx + dy * dy <= self.cfg.circle_radius ** 2

    def get_dirichlet_value_raw(self, i: int, j: int) -> Optional[float]:
        """
        Retorna a temperatura se o nó pertencer a alguma fronteira ou à inclusão circular.
        Caso retorne None, o nó é um ponto interno onde a EDP deverá ser resolvida.
        """
        x = self.x[i]
        y = self.y[j]

        # Bordas
        if i == 0:
            return float(self.cfg.TL(y))
        if i == self.cfg.Nx - 1:
            return float(self.cfg.TR(y))
        if j == 0:
            return float(self.cfg.TB(x))
        if j == self.cfg.Ny - 1:
            return float(self.cfg.TT(x))

        # Obstáculo Interno
        if self.is_inside_circle_constraint(x, y):
            return float(self.cfg.TC)

        return None

    def get_dirichlet_value(self, i: int, j: int) -> Optional[float]:
        return self.get_dirichlet_value_raw(i, j)

    def assembly(self, matrix_mode: str = "sparse") -> Union[np.ndarray, sparse.csr_matrix]:
        """
        Montagem da matriz de coeficientes 'A' utilizando formulação conservativa de FVM.

        Intuição Física (Balanço Térmico Nodal):
        Em estado estacionário, a soma dos fluxos de calor pelas faces de um volume de controle 
        deve ser igual à geração interna.
        Fluxo através da face Leste (Lei de Fourier aproximada):
        $$ q_e \approx -k_e \frac{T_E - T_P}{hx} \Delta y $$

        Risco/Melhoria Técnica:
        O código interpola k nas faces (ex: xc + 0.5*hx). Se houver um salto abrupto na 
        condutividade (ex: interface entre dois materiais muito diferentes), avaliar k no 
        ponto médio não conserva o fluxo adequadamente. A literatura de FVM recomenda o uso da 
        Média Harmônica para k_face (Patankar, 1980). Dado o problema proposto no PDF, o k
        modificado parece ser suave, então a interpolação direta é aceitável, mas é um ponto 
        de fragilidade arquitetural do solver.
        """
        if matrix_mode not in ("dense", "sparse"):
            raise ValueError("matrix_mode deve ser 'dense' ou 'sparse'.")

        # Arrays para montagem no formato COO (Coordinate), eficiente para construir a matriz.
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        # Razões de aspecto do volume de controle (área de fluxo / distância nodal)
        rx = self.hy / self.hx
        ry = self.hx / self.hy

        for j in range(self.cfg.Ny):
            for i in range(self.cfg.Nx):
                Ic = ij2n(i, j, self.cfg.Nx)

                # Se for um nó de contorno, a equação é trivialmente T_i = valor.
                # A matriz recebe 1 na diagonal principal.
                if Ic in self.dirichlet_set:
                    rows.append(Ic)
                    cols.append(Ic)
                    data.append(1.0)
                    continue

                Ie = ij2n(i + 1, j, self.cfg.Nx)
                Iw = ij2n(i - 1, j, self.cfg.Nx)
                In = ij2n(i, j + 1, self.cfg.Nx)
                Is = ij2n(i, j - 1, self.cfg.Nx)

                xc = self.x[i]
                yc = self.y[j]

                # Condutividade térmica calculada no centro das faces (East, West, North, South)
                ke = self.get_k(xc + 0.5 * self.hx, yc)
                kw = self.get_k(xc - 0.5 * self.hx, yc)
                kn = self.get_k(xc, yc + 0.5 * self.hy)
                ks = self.get_k(xc, yc - 0.5 * self.hy)

                # Coeficientes da equação linear: a_P * T_P = a_E*T_E + a_W*T_W + a_N*T_N + a_S*T_S + b
                # A formulação implícita aqui passa tudo pro mesmo lado, logo os vizinhos são negativos na matriz.
                aE = ke * rx
                aW = kw * rx
                aN = kn * ry
                aS = ks * ry
                aP = aE + aW + aN + aS

                rows.extend([Ic, Ic, Ic, Ic, Ic])
                cols.extend([Ic, Ie, Iw, In, Is])
                data.extend([aP, -aE, -aW, -aN, -aS])

        # A matriz é instanciada e convertida para CSR (Compressed Sparse Row),
        # formato mais rápido para multiplicação matriz-vetor e resolução de sistemas (spsolve).
        A_sparse = sparse.coo_matrix((data, (rows, cols)), shape=(self.nunk, self.nunk)).tocsr()
        if matrix_mode == "sparse":
            return A_sparse
        return A_sparse.toarray()

    def build_rhs(self) -> np.ndarray:
        """
        Monta o vetor independente 'b' (Right-Hand Side).
        A contribuição de um termo fonte volumétrico S_P em um volume V_P é S_P * V_P.
        No 2D, considerando espessura unitária, o volume é apenas a área hx * hy.
        """
        b = np.zeros(self.nunk, dtype=float)
        for j in range(self.cfg.Ny):
            for i in range(self.cfg.Nx):
                Ic = ij2n(i, j, self.cfg.Nx)
                if Ic in self.dirichlet_set:
                    b[Ic] = self.dirichlet_value_by_id[Ic]
                else:
                    b[Ic] = self.get_source(self.x[i], self.y[j]) * self.hx * self.hy
        return b

    def apply_dirichlet_bc_sparse(self, A: sparse.csr_matrix, b: np.ndarray) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        [DEPRECIADO NO FLUXO PRINCIPAL - MANTIDO POR COMPATIBILIDADE]
        Impõe T = T_prescrita manipulando a matriz já montada.

        Fundamento `preserve_symmetry=True`:
        Se a matriz A originalmente simétrica for alterada na linha i zerando todos os elementos
        (exceto a diagonal), a coluna i continuará com valores não-nulos, destruindo a simetria.
        Para preservar a simetria de A, pegamos o valor da coluna, passamos subtraindo para o vetor 
        'b', e então zeramos tanto a linha quanto a coluna i.
        """
        A_mod = A.copy().tolil() # LIL é mais eficiente que CSR para alterar fatiamentos/esparsidade.
        b_mod = b.copy()
        for Ic, Tpresc in zip(self.dirichlet_ids, self.dirichlet_values):
            if self.cfg.preserve_symmetry:
                col = A_mod[:, Ic].toarray().ravel()
                b_mod -= col * Tpresc
                A_mod[:, Ic] = 0.0
                A_mod[Ic, :] = 0.0
                A_mod[Ic, Ic] = 1.0
                b_mod[Ic] = Tpresc
            else:
                A_mod[Ic, :] = 0.0
                A_mod[Ic, Ic] = 1.0
                b_mod[Ic] = Tpresc
        return A_mod.tocsr(), b_mod

    def apply_dirichlet_bc_dense(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Análogo ao caso esparso, porém em matriz NumPy nativa."""
        A_mod = A.copy()
        b_mod = b.copy()
        for Ic, Tpresc in zip(self.dirichlet_ids, self.dirichlet_values):
            if self.cfg.preserve_symmetry:
                b_mod -= A_mod[:, Ic] * Tpresc
                A_mod[:, Ic] = 0.0
                A_mod[Ic, :] = 0.0
                A_mod[Ic, Ic] = 1.0
                b_mod[Ic] = Tpresc
            else:
                A_mod[Ic, :] = 0.0
                A_mod[Ic, Ic] = 1.0
                b_mod[Ic] = Tpresc
        return A_mod, b_mod

    def solve_system(self, matrix_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Orquestra a montagem (Assembly) e a solução do sistema linear.
        Retorna um dicionário com os campos calculados e tempos de execução (profiling).
        """
        if matrix_mode is None:
            matrix_mode = self.cfg.solver_mode

        t0 = time.perf_counter()
        A = self.assembly(matrix_mode=matrix_mode)
        b = self.build_rhs()
        t1 = time.perf_counter()

        # Decisão de Projeto Crítica apontada no comentário original:
        # A montagem (assembly) já insere as linhas do contorno Dirichlet diretamente (1.0 na diagonal).
        # Logo, o passo de "alterar a matriz à posteriori" foi bypassado para ganhar desempenho.
        A_mod, b_mod = A, b
        
        if matrix_mode == "dense":
            T_vec = self.solve_linear_system_dense(A_mod, b_mod)
        else:
            T_vec = self.solve_linear_system_sparse(A_mod, b_mod)
        t2 = time.perf_counter()

        T_grid = T_vec.reshape((self.cfg.Ny, self.cfg.Nx))
        result = {
            "A": A,
            "A_mod": A_mod,
            "b": b,
            "b_mod": b_mod,
            "T_vec": T_vec,
            "T_grid": T_grid,  # Formato necessário para plotar via Contourf
            "x": self.x.copy(),
            "y": self.y.copy(),
            "Tmax": float(np.max(T_grid)),
            "Tmean": float(np.mean(T_grid)),
            "assembly_time": t1 - t0,
            "solve_time": t2 - t1,
            "total_time": t2 - t0,
        }
        self.last_result = result
        return result

# ============================================================
# FUNÇÕES HIDRÁULICAS
# ============================================================

def water_viscosity_pa_s(T_celsius: float) -> float:
    """
    Função preservada do código original (Modelo de Vogel-Fulcher-Tammann aproximado).
    Trade-off: Embora fisicamente válida para uma ampla faixa, no contexto deste
    acoplamento específico (Capítulo 4), ela foi substituída por `empirical_viscosity`
    para garantir paridade numérica com o gabarito do PDF.
    """
    A = 2.414e-5
    B = 247.8
    C = 140.0
    return A * 10 ** (B / ((T_celsius + 273.15) - C))


def empirical_viscosity(T_celsius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Viscosidade dinâmica do fluido (agua) definida no PDF.
    Formula: mu(T) = 0.001791 / (1 + 0.03368*T + 0.000221*T^2)
    
    Vetorização: O uso de np.asarray permite que a função receba tanto um escalar (float)
    quanto um vetor de temperaturas (numpy array). Isso evita laços "for" custosos 
    ao calcular a viscosidade em todas as arestas simultaneamente.
    """
    T = np.asarray(T_celsius, dtype=float)
    mu = 0.001791 / (1.0 + 0.03368 * T + 0.000221 * T ** 2)
    if np.isscalar(T_celsius):
        return float(mu)
    return mu


def rectangular_area(width: float, height: float) -> float:
    """Verificação básica de sanidade (Sanity Check) para dimensões físicas."""
    if width <= 0 or height <= 0:
        raise ValueError("Largura e altura devem ser positivas.")
    return width * height


def equivalent_diameter_from_area(area: float) -> float:
    """
    Calcula o diâmetro de um círculo com a mesma área transversal do canal.
    """
    if area <= 0:
        raise ValueError("A área deve ser positiva.")
    return float(np.sqrt(4.0 * area / np.pi))


def edge_lengths(Xno: np.ndarray, conec: np.ndarray) -> np.ndarray:
    """
    Comprimento euclidiano de cada aresta/canal (L_k).
    Utiliza operações matriciais (axis=1) para calcular a norma de todos os vetores
    (p1 - p0) de uma vez, garantindo alta performance computacional.
    """
    p0 = Xno[conec[:, 0], :]
    p1 = Xno[conec[:, 1], :]
    return np.linalg.norm(p1 - p0, axis=1)


def build_incidence_matrix(conec: np.ndarray, nv: Optional[int] = None) -> np.ndarray:
    """
    Matriz de incidência orientada D do grafo (Arestas x Nós).
    
    Fundamento:
    Mapeia a topologia da rede. Se a aresta k conecta o nó i ao nó j:
        D[k, i] =  1 (fluxo sai de i)
        D[k, j] = -1 (fluxo entra em j)
    
    Essa matriz é o operador de diferença (gradiente discreto) do grafo. 
    A queda de pressão nas arestas é calculada por: delta_p = D * p.
    """
    if nv is None:
        nv = int(np.max(conec)) + 1
    D = np.zeros((conec.shape[0], nv), dtype=float)
    for k, (i, j) in enumerate(conec.astype(int)):
        D[k, i] = 1.0
        D[k, j] = -1.0
    return D


def assembly(conec: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Monta a matriz global hidráulica A (Matriz Laplaciana Ponderada do Grafo).

    Fundamento Matemático:
    Conservação de massa (Lei dos Nós de Kirchhoff): A soma dos fluxos em um nó é zero.
    Como q_k = C_k * (p_i - p_j), a matriz resultante A tem a propriedade algébrica:
    A = D^T * K * D, onde K é a matriz diagonal de condutâncias.
    
    Construção Direta:
    A matriz é montada adicionando a condutância C_k na diagonal dos nós i e j, 
    e subtraindo nas posições fora da diagonal (i,j) e (j,i).
    """
    nv = int(np.max(conec)) + 1
    A = np.zeros((nv, nv), dtype=float)
    for k, (i, j) in enumerate(conec.astype(int)):
        ck = float(C[k])
        A[i, i] += ck
        A[j, j] += ck
        A[i, j] -= ck
        A[j, i] -= ck
    return A


def evaluate_flow_spec(spec: dict, t: float) -> float:
    """
    Avalia a vazão prescrita no tempo t.
    Decisão de Projeto: Aceita especificações em dicionário para permitir fluxos 
    transientes (senoidais), embora o acoplamento do Cap. 4 seja resolvido no regime estacionário (t=0).
    """
    flow_type = spec["type"].lower()
    if flow_type == "constant":
        return float(spec["value"])
    if flow_type == "sin":
        return float(spec["mean"] + spec["amp"] * np.sin(2.0 * np.pi * spec["freq"] * t + spec.get("phase", 0.0)))
    if flow_type == "cos":
        return float(spec["mean"] + spec["amp"] * np.cos(2.0 * np.pi * spec["freq"] * t + spec.get("phase", 0.0)))
    raise ValueError(f"Tipo de vazão inválido: {flow_type}")


def evaluate_flow_bc(flow_bc: dict, t: float, nv: int) -> np.ndarray:
    """Monta o vetor independente de vazões nodais (condições de Neumann)."""
    b = np.zeros(nv, dtype=float)
    for node, spec in flow_bc.items():
        if 0 <= int(node) < nv:
            b[int(node)] += evaluate_flow_spec(spec, t)
    return b


def apply_pressure_bc(A: np.ndarray, b: np.ndarray, pressure_bc: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Impõe pressões prescritas (condição de Dirichlet no grafo).
    Risco: A Laplaciana A é naturalmente singular (determinante nulo), pois pressões 
    relativas não definem um campo absoluto. Fixar ao menos uma pressão (ex: p_out = 0)
    quebra a singularidade e permite a inversão da matriz.
    """
    if len(pressure_bc) == 0:
        raise ValueError("É necessário prescrever pelo menos uma pressão nodal para evitar matriz singular.")
    A_mod = A.copy()
    b_mod = b.copy()
    for node, p_value in pressure_bc.items():
        node = int(node)
        A_mod[node, :] = 0.0
        A_mod[node, node] = 1.0
        b_mod[node] = float(p_value)
    return A_mod, b_mod


def get_area_per_edge(conec: np.ndarray, cfg: dict) -> np.ndarray:
    """Recupera a área de seção transversal de cada microcanal."""
    nc = conec.shape[0]
    if cfg.get("area_per_edge") is not None:
        area_edge = np.array(cfg["area_per_edge"], dtype=float)
        if len(area_edge) != nc:
            raise ValueError("area_per_edge deve ter o mesmo tamanho do número de arestas.")
        return area_edge
    area = cfg.get("area_constant", cfg.get("width", 500e-6) * cfg.get("height", 500e-6))
    return np.full(nc, float(area), dtype=float)


def hydraulic_conductivities(Xno: np.ndarray, conec: np.ndarray, cfg: dict) -> dict:
    """
    Calcula a condutância hidráulica (C) de cada canal.

    Fundamento / Limitação (Hagen-Poiseuille):
    A equação kappa = (pi * D_eq^4) / (128 * mu) assume escoamento laminar
    plenamente desenvolvido em um duto circular. O uso de um "diâmetro equivalente"
    para dutos retangulares/quadrados (500x500 µm) é uma aproximação de engenharia.
    Para seções não-circulares rigorosas, fatores de forma (shape factors) baseados
    em séries infinitas ou simulações Navier-Stokes seriam necessários.
    """
    L = edge_lengths(Xno, conec)
    area_edge = get_area_per_edge(conec, cfg)
    D_eq = np.sqrt(4.0 * area_edge / np.pi)

    # A injeção da temperatura no fluido ocorre aqui: a viscosidade nas arestas
    # é passada no dicionário cfg via 'mu_per_edge'.
    if "mu_per_edge" in cfg and cfg["mu_per_edge"] is not None:
        mu = np.array(cfg["mu_per_edge"], dtype=float)
        if len(mu) != len(L):
            raise ValueError("mu_per_edge deve ter o mesmo tamanho do número de arestas.")
    else:
        T_global = float(cfg.get("temperature_celsius", 25.0))
        mu = np.full_like(L, empirical_viscosity(T_global), dtype=float)

    kappa = np.pi * D_eq ** 4 / (128.0 * mu)
    C = kappa / L  # Condutância = Permeabilidade do canal / Comprimento

    return {
        "mu": mu,
        "lengths": L,
        "area_edge": area_edge,
        "diameter_eq_edge": D_eq,
        "kappa_edge": kappa,
        "conductance_edge": C,
    }


def solve_network(conec: np.ndarray, C: np.ndarray, pressure_bc: dict, flow_bc: dict, t: float) -> dict:
    """
    Resolve a rede hidráulica estacionária: A * p = b.
    Recupera as vazões nas arestas q através de q = C * delta_p.
    """
    A = assembly(conec, C)
    nv = A.shape[0]
    b = evaluate_flow_bc(flow_bc, t, nv)
    A_mod, b_mod = apply_pressure_bc(A, b, pressure_bc)

    # Resolve o sistema linear denso da rede hidráulica.
    p = np.linalg.solve(A_mod, b_mod)

    D = build_incidence_matrix(conec, nv)
    K = np.diag(C)
    dp_edge = D @ p
    q = C * dp_edge  # Lei de Poiseuille em notação vetorial

    return {
        "A": A,
        "A_mod": A_mod,
        "b": b,
        "b_mod": b_mod,
        "p": p,
        "D": D,
        "K": K,
        "dp_edge": dp_edge,
        "q": q,
    }


def compute_power(p: np.ndarray, D: np.ndarray, K: np.ndarray) -> float:
    """
    Cálculo da potência de bombeamento do fluido.
    P = Potência = dp^T * K * dp = sum(C_k * (p_i - p_j)^2)
    É a taxa de energia dissipada pelo atrito viscoso.
    """
    dp = D @ p
    return float(dp.T @ K @ dp)


def nodal_mass_residual(conec: np.ndarray, q: np.ndarray, imposed_b: np.ndarray) -> np.ndarray:
    """
    Verificação de estabilidade (Balanço de Massa).
    Calcula a diferença entre o fluxo que entra, o fluxo que sai e a fonte nodal.
    Valores muito diferentes da tolerância da máquina (e.g., 1e-15) indicariam bugs
    na montagem de A ou no solver linear.
    """
    nv = len(imposed_b)
    residual = -imposed_b.copy()
    for k, (i, j) in enumerate(conec.astype(int)):
        residual[i] += q[k]
        residual[j] -= q[k]
    return residual


def print_inputs_summary(cfg: dict, hydraulic_data: dict, Xno: np.ndarray, conec: np.ndarray) -> None:
    """Resumo em CLI preservado da parte hidráulica."""
    print("\n" + "=" * 88)
    print("ENTRADAS DA REDE HIDRÁULICA")
    print("=" * 88)
    print(f"nós = {Xno.shape[0]}, arestas = {conec.shape[0]}")
    print(f"área média = {np.mean(hydraulic_data['area_edge']):.6e} m²")
    print(f"comprimento médio = {np.mean(hydraulic_data['lengths']):.6e} m")
    print(f"mu média = {np.mean(hydraulic_data['mu']):.6e} Pa.s")
    print("pressões prescritas:", cfg.get("pressure_bc", {}))
    print("vazões prescritas:", cfg.get("flow_bc", {}))
    print("=" * 88)


def print_output_summary(cfg: dict, hydraulic_data: dict, result: dict, conec: np.ndarray) -> None:
    """Resumo de saída e diagnóstico numérico em CLI."""
    p = result["p"]
    q = result["q"]
    power = compute_power(p, result["D"], result["K"])
    residual = nodal_mass_residual(conec, q, result["b"])
    print("\n" + "=" * 88)
    print("SAÍDAS DA REDE HIDRÁULICA")
    print("=" * 88)
    print(f"pressão máxima = {np.max(p):.6e} Pa")
    print(f"pressão mínima = {np.min(p):.6e} Pa")
    print(f"potência total = {power:.6e} W")
    print(f"máximo resíduo de massa = {np.max(np.abs(residual)):.6e} m³/s")
    print("=" * 88)


def print_final_explanation(cfg: dict, hydraulic_data: dict, result: dict) -> None:
    """Explicação da física acoplada do sistema."""
    print("\nINTERPRETAÇÃO FINAL")
    print("- A temperatura altera mu(T), que altera a condutância C_k e, por isso, redistribui as pressões.")
    print("- Como a viscosidade mu(T) de líquidos diminui quando T aumenta, canais mais quentes tendem a apresentar menor resistência hidráulica e conduzir maiores vazões.")
    print("- A potência total é a energia dissipada pelo sistema devido à fricção viscosa, calculada por soma de C_k*(Δp_k)^2 em todas as arestas.")

# ============================================================
# GERAÇÃO DA REDE HIDRÁULICA
# ============================================================

# Programação Defensiva: Tenta importar os scripts externos do docente.
# Caso os arquivos não estejam no mesmo diretório, o script não quebra (evita ImportError),
# sinalizando a falha e utilizando a função geradora substituta (fallback).
try:
    from gera_grafo import generate_graph_arrays as _external_generate_graph_arrays
    print(">> SUCESSO: 'gera_grafo.py' do professor foi importado!")
except Exception as e:
    print(f"\n[ERRO] Não foi possível carregar 'gera_grafo.py': {e}")
    _external_generate_graph_arrays = None

try:
    from plota_rede import PlotaRede
    print(">> SUCESSO: 'plota_rede.py' do professor foi importado!")
except Exception as e:
    print(f"\n[ERRO] Não foi possível carregar 'plota_rede.py': {e}")
    PlotaRede = None


def generate_graph_arrays(complex_level: int = 3, spine_length: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper de compatibilidade para a topologia da rede.

    Intuição:
    O modelo precisa das coordenadas nodais e da matriz de conectividade (lista de arestas).
    Se o gerador original falhar, instanciamos um grafo retangular simples que atende aos
    requisitos de contorno numérico do problema (nós 0 e 175 nas extremidades).
    """
    if _external_generate_graph_arrays is not None:
        return _external_generate_graph_arrays(complex_level)

    # Gerador substituto: malha de 16x11 = 176 nós (índices 0 a 175).
    nx_nodes = 16
    ny_nodes = 11
    x = np.linspace(0.0, (spine_length - 1) * 4.0, nx_nodes)
    y = np.linspace(-5.0, 5.0, ny_nodes)

    X: List[Tuple[float, float]] = []
    for jj in range(ny_nodes):
        for ii in range(nx_nodes):
            X.append((x[ii], y[jj]))
    Xno = np.array(X, dtype=float)

    conec: List[Tuple[int, int]] = []
    for jj in range(ny_nodes):
        for ii in range(nx_nodes):
            n = ii + jj * nx_nodes
            # Conexões horizontais (i, i+1) e verticais (i, i+nx)
            if ii + 1 < nx_nodes:
                conec.append((n, n + 1))
            if jj + 1 < ny_nodes:
                conec.append((n, n + nx_nodes))
    return Xno, np.array(conec, dtype=int)


def build_pdf_network(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Constrói a rede, converte unidades e aplica translações espaciais.

    Decisão de Projeto:
    A separação desta função isola a lógica de tratamento de dados espaciais
    (conversão mm -> m e centralização $y$) da lógica puramente topológica.
    """
    Xno, conec = generate_graph_arrays(cfg["levels"], cfg["spine_length"])
    Xno = np.asarray(Xno, dtype=float) * cfg["coord_scale_to_m"]
    conec = np.asarray(conec, dtype=int)

    # Translação geométrica: move o eixo y da rede para embuti-la no centro da placa térmica.
    Xno[:, 1] += 0.5 * cfg["Ly"]

    # Busca espacial pelo nó de saída: encontra o índice cujo vetor posição minimiza a
    # distância euclidiana em relação à coordenada teórica de saída.
    xout = cfg["coord_scale_to_m"] * (cfg["spine_length"] - 1) * 4.0
    target = np.array([xout, 0.5 * cfg["Ly"]])
    outlet_node = int(np.argmin(np.linalg.norm(Xno - target, axis=1)))
    return Xno, conec, outlet_node


# ============================================================
# INTERPOLAÇÃO BIDIMENSIONAL DE TEMPERATURA
# ============================================================

class TemperatureInterpolator:
    """
    Mapeia o campo térmico discreto $T_{i,j}$ para o domínio contínuo $T(x,y)$.

    Fundamento:
    O modelo térmico e o modelo hidráulico operam em espaços discretos diferentes
    (malha estruturada vs. grafo não-estruturado). Para passar dados térmicos para
    a rede, é necessária uma função de transferência espacial.

    Trade-offs:
    - "linear" (Bilinear): Rápida, não cria oscilações espúrias, mas não preserva
      a continuidade da primeira derivada (gradiente em degraus).
    - "cubic" (Bicúbica): Suave (preserva derivadas), mas computacionalmente mais cara e 
      sujeita a "overshoots" (fenômeno de Runge) próximos a grandes gradientes.
    - "nearest" (Vizinho mais próximo): Baixa ordem ($O(h)$), gera campo constante em blocos,
      útil apenas para diagnósticos ou funções descontínuas.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, T_grid: np.ndarray, method: str = "linear") -> None:
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.T_grid = np.asarray(T_grid, dtype=float)
        self.method = method.lower()

        if self.method in ("linear", "nearest"):
            # bounds_error=False previne interrupções da simulação se, por erro de arredondamento,
            # um nó da rede cair microscopicalmente fora da borda da placa térmica.
            self._interp = RegularGridInterpolator(
                (self.x, self.y), self.T_grid.T,
                method=self.method,
                bounds_error=False,
                fill_value=None,
            )
            self._spline = None
        elif self.method == "cubic":
            self._interp = None
            # kx=3, ky=3 define o grau 3 dos polinômios (Spline cúbica).
            self._spline = RectBivariateSpline(self.x, self.y, self.T_grid.T, kx=3, ky=3)
        else:
            raise ValueError("method deve ser 'linear', 'nearest' ou 'cubic'.")

    def __call__(self, pts_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_xy, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        if self.method in ("linear", "nearest"):
            return np.asarray(self._interp(pts), dtype=float)
        return np.asarray(self._spline.ev(pts[:, 0], pts[:, 1]), dtype=float)


# ============================================================
# QUADRATURAS EM ARESTAS DA REDE
# ============================================================

def edge_quadrature_points(p0: np.ndarray, p1: np.ndarray, rule: str, subdivisions: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera pontos e pesos para integração numérica unidimensional (Fórmulas de Newton-Cotes).

    Fundamento:
    Aproxima $\int_0^1 f(s) ds \approx \sum_{i=1}^N w_i f(s_i)$.
    Para 'midpoint' e 'trapezoid', o erro de truncamento é $O(h^2)$, onde $h = 1/N$.
    """
    if subdivisions < 1:
        raise ValueError("subdivisions deve ser >= 1.")

    rule = rule.lower()
    if rule == "midpoint":
        # Avalia no centro de cada subsegmento.
        alpha = (np.arange(subdivisions) + 0.5) / subdivisions
        weights = np.full(subdivisions, 1.0 / subdivisions, dtype=float)
    elif rule == "trapezoid":
        # Avalia nos limites. Requer N+1 pontos para N subdivisões.
        alpha = np.linspace(0.0, 1.0, subdivisions + 1)
        weights = np.full(subdivisions + 1, 1.0 / subdivisions, dtype=float)
        weights[0] *= 0.5
        weights[-1] *= 0.5
    else:
        raise ValueError("rule deve ser 'midpoint' ou 'trapezoid'.")

    # Equação paramétrica do segmento de reta: p(a) = p0 + a*(p1 - p0)
    pts = p0[None, :] + alpha[:, None] * (p1 - p0)[None, :]
    return pts, weights


def mean_edge_temperature(
    Xno: np.ndarray,
    conec: np.ndarray,
    interpolator: TemperatureInterpolator,
    rule: str,
    subdivisions: int,
) -> np.ndarray:
    """
    Calcula a temperatura média em cada aresta: $\langle T_k \rangle = \frac{1}{L_k} \int_0^{L_k} T(s) ds$.
    """
    values = np.zeros(conec.shape[0], dtype=float)
    for k, (i, j) in enumerate(conec.astype(int)):
        pts, weights = edge_quadrature_points(Xno[i], Xno[j], rule, subdivisions)
        T_pts = interpolator(pts)
        values[k] = float(np.sum(weights * T_pts))
    return values


def mean_edge_viscosity_direct(
    Xno: np.ndarray,
    conec: np.ndarray,
    interpolator: TemperatureInterpolator,
    rule: str,
    subdivisions: int,
) -> np.ndarray:
    """
    Integração direta da viscosidade: $\langle \mu_k \rangle = \frac{1}{L_k} \int_0^{L_k} \mu(T(s)) ds$.

    Fundamento:
    Pela Desigualdade de Jensen, se $\mu(T)$ é estritamente convexa ou côncava,
    então $\mu(\langle T \rangle) \neq \langle \mu(T) \rangle$. Como a viscosidade da água cai
    exponencialmente com a temperatura, o comportamento não é linear, e integrar
    a propriedade diretamente reflete com mais fidelidade a resistência hidrodinâmica total.
    """
    values = np.zeros(conec.shape[0], dtype=float)
    for k, (i, j) in enumerate(conec.astype(int)):
        pts, weights = edge_quadrature_points(Xno[i], Xno[j], rule, subdivisions)
        T_pts = interpolator(pts)
        values[k] = float(np.sum(weights * empirical_viscosity(T_pts)))
    return values


# ============================================================
# DISTÂNCIA PONTO-SEGMENTO E INFLUÊNCIA DA REDE NA PLACA
# ============================================================

def point_segment_distance_batch(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cálculo analítico vetorizado da menor distância entre pontos $P$ e um segmento $\overline{AB}$.

    Fundamento Matemático:
    1. Define-se a reta paramétrica $P_{proj}(t) = A + t(B - A)$.
    2. Encontra-se o parâmetro $t$ através da projeção ortogonal do vetor $\vec{PA}$ sobre $\vec{AB}$:
       $$ t = \frac{(\vec{P} - \vec{A}) \cdot (\vec{B} - \vec{A})}{\|\vec{B} - \vec{A}\|^2} $$
    3. Se $t \notin [0,1]$, a projeção cai fora do segmento. A função `np.clip` trunca
       $t$ para 0 (ponto $A$) ou 1 (ponto $B$), resolvendo esse caso de borda.
    """
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        # Segmento degenerado (A e B são o mesmo ponto).
        return np.linalg.norm(points - a, axis=1)
    t = ((points - a) @ ab) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - closest, axis=1)


class NetworkInfluenceModel:
    """
    Modelo geométrico de proximidade para Acoplamento via Campo Modificado.

    Fundamento Estrutural (KD-Tree):
    Para cada nó da malha térmica (Nós Térmicos = $N$), precisamos verificar a distância
    a todas as arestas da rede (Arestas = $M$). Uma abordagem força-bruta custa $O(N \times M)$.
    Ao indexar os centros das arestas usando uma k-d tree, a busca por arestas dentro de um
    raio de influência reduz o custo assintótico da consulta para $O(\log M)$, otimizando severamente
    o tempo de simulação para malhas refinadas.
    """

    def __init__(self, Xno: np.ndarray, conec: np.ndarray, d_max: float, k0: float = 0.25) -> None:
        self.Xno = np.asarray(Xno, dtype=float)
        self.conec = np.asarray(conec, dtype=int)
        self.d_max = float(d_max)
        self.k0 = float(k0)

        # Pré-processamento dos centros geométricos.
        self.edge_a = self.Xno[self.conec[:, 0]]
        self.edge_b = self.Xno[self.conec[:, 1]]
        self.edge_mid = 0.5 * (self.edge_a + self.edge_b)
        self.edge_half_length = 0.5 * np.linalg.norm(self.edge_b - self.edge_a, axis=1)
        
        # Constrói a estrutura de particionamento espacial.
        self.tree = cKDTree(self.edge_mid)

        # Heurística conservadora de busca: d_max (raio alvo) + metade do comprimento da aresta.
        # Garante que arestas compridas cujas extremidades tangenciam a vizinhança não sejam ignoradas.
        self.search_radius = self.d_max + float(np.max(self.edge_half_length))

    def nearby_edges_with_distances(self, point: np.ndarray) -> List[Tuple[int, float]]:
        """Realiza a consulta topológica retornando a tupla (ID_aresta, distância)."""
        candidate_ids = self.tree.query_ball_point(point, self.search_radius)
        out: List[Tuple[int, float]] = []
        p = np.asarray(point, dtype=float).reshape(1, 2)
        for eid in candidate_ids:
            d = point_segment_distance_batch(p, self.edge_a[eid], self.edge_b[eid])[0]
            if d <= self.d_max:
                out.append((int(eid), float(d)))
        return out

    def CreateMapDistance(self, Lx: float, Ly: float, Nx: int, Ny: int) -> Dict[int, List[Tuple[int, float]]]:
        """Itera sobre a malha térmica construindo um dicionário estático de distâncias."""
        x = np.linspace(0.0, Lx, Nx)
        y = np.linspace(0.0, Ly, Ny)
        mapa: Dict[int, List[Tuple[int, float]]] = {}
        for j, yy in enumerate(y):
            for i, xx in enumerate(x):
                idx = ij2n(i, j, Nx)
                mapa[idx] = self.nearby_edges_with_distances(np.array([xx, yy]))
        return mapa

    def k_modified_at_point(self, x: float, y: float) -> float:
        """
        Calcula a condutividade térmica modificada $k_f(x,y)$.
        A condutividade é aumentada artificialmente nas imediações dos microcanais
        (representando o aprimoramento na troca de calor gerado pelo escoamento):
        $$ k_f = k_0 \left(1 + \sum_{j \in V_f} \frac{1}{1+d_j}\right) $$
        """
        prox = self.nearby_edges_with_distances(np.array([x, y], dtype=float))
        return self.k0 * (1.0 + sum(1.0 / (1.0 + d) for _, d in prox))

    def source_extra_at_point(self, x: float, y: float, S0: float, intensity_edge: np.ndarray) -> float:
        """
        Calcula a perturbação de energia (sorvedouro ou fonte).
        O modelo assume uma dispersão Gaussiana em torno das arestas para evitar singularidades
        (funções delta de Dirac) na malha de volumes finitos:
        $$ S_p = S_0 \sum I_j \exp\left(-\frac{d_j^2}{2\sigma^2}\right) $$
        com $\sigma = d_{max}/2$.
        """
        prox = self.nearby_edges_with_distances(np.array([x, y], dtype=float))
        if not prox:
            return 0.0
        sigma = self.d_max / 2.0
        total = 0.0
        for eid, d in prox:
            total += intensity_edge[eid] * np.exp(-(d ** 2) / (2.0 * sigma ** 2))
        return float(S0 * total)


def CreateMapDistance(Lx: float, Ly: float, Nx: int, Ny: int, Xno: np.ndarray, conec: np.ndarray, d_max: float) -> Dict[int, List[Tuple[int, float]]]:
    """Encapsulamento de compatibilidade global requerido pela assinatura do PDF."""
    return NetworkInfluenceModel(Xno, conec, d_max).CreateMapDistance(Lx, Ly, Nx, Ny)

# ============================================================
# CLASSE DE ACOPLAMENTO HIDRÁULICO-TÉRMICO
# ============================================================

class HydroThermalCoupledModel:
    """
    Orquestrador (Padrão de Projeto Facade).
    Organiza a integração "one-way" (ou fluxo sequencial) entre a placa térmica 
    e a rede hidráulica, abstraindo a complexidade das classes individuais.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        # Inicializa a topologia e geometria da rede assim que o modelo é instanciado.
        self.Xno, self.conec, self.outlet_node = build_pdf_network(cfg)

    def make_thermal_config(self, Nx: int, Ny: int, *, k_function: Optional[FieldFunction2D] = None,
                            source_function: Optional[FieldFunction2D] = None) -> ThermalPlateConfig:
        """
        Fábrica de configurações térmicas.
        Gera as condições de contorno padrão do PDF para qualquer resolução de malha (Nx, Ny),
        permitindo injetar k(x,y) e S(x,y) modificados pela rede.
        """
        return ThermalPlateConfig(
            Lx=self.cfg["Lx"],
            Ly=self.cfg["Ly"],
            Nx=Nx,
            Ny=Ny,
            TL=self.cfg["TL"],
            TR=self.cfg["TR"],
            TB=top_bottom_temperature_function(self.cfg["Lx"]),
            TT=top_bottom_temperature_function(self.cfg["Lx"]),
            source_function=source_function or constant_source(self.cfg["source_value"]),
            use_variable_k=(k_function is not None),
            k_constant=self.cfg["k_constant"],
            k_function=k_function,
            use_circle_constraint=True,
            circle_center_x=self.cfg["circle_center_x"],
            circle_center_y=self.cfg["circle_center_y"],
            circle_radius=self.cfg["circle_radius"],
            TC=self.cfg["TC"],
            solver_mode="sparse",
            output_dir=OUTPUT_DIR,
        )

    def solve_thermal(self, Nx: int, Ny: int, *, k_function: Optional[FieldFunction2D] = None,
                      source_function: Optional[FieldFunction2D] = None) -> Dict[str, Any]:
        """Instancia e resolve a placa térmica isoladamente, retornando o dicionário de resultados."""
        cfg = self.make_thermal_config(Nx, Ny, k_function=k_function, source_function=source_function)
        solver = ThermalPlateSolver(cfg)
        return solver.solve_system()

    def hydraulic_cfg_from_mu(self, mu_per_edge: np.ndarray) -> Dict[str, Any]:
        """
        Gera o dicionário de configuração da rede hidráulica.
        A "cola" do acoplamento acontece aqui: recebe as viscosidades calculadas
        pela malha térmica e as injeta no solver de escoamento.
        """
        return {
            "width": self.cfg["width"],
            "height": self.cfg["height"],
            "area_constant": self.cfg["area_constant"],
            "mu_per_edge": mu_per_edge,
            "pressure_bc": {self.outlet_node: 0.0}, # Pressão de referência (zero) na saída
            "flow_bc": {
                self.cfg["inlet_0"]: {"type": "constant", "value": self.cfg["Q0_in"]},
                self.cfg["inlet_175"]: {"type": "constant", "value": self.cfg["Q175_in"]},
            },
        }

    def solve_hydraulic_from_edge_temperatures(self, T_edge_mean: np.ndarray) -> Dict[str, Any]:
        """
        Fluxo Clássico: Calcula a viscosidade a partir da Temperatura Média da aresta ( mu(<T>) ).
        Atualiza as condutâncias e resolve as pressões.
        """
        mu_edge = empirical_viscosity(T_edge_mean)
        hcfg = self.hydraulic_cfg_from_mu(mu_edge)
        hydraulic_data = hydraulic_conductivities(self.Xno, self.conec, hcfg)
        result = solve_network(self.conec, hydraulic_data["conductance_edge"], hcfg["pressure_bc"], hcfg["flow_bc"], t=0.0)
        return {
            "hydraulic_cfg": hcfg,
            "hydraulic_data": hydraulic_data,
            "result": result,
            "pmax": float(np.max(result["p"])),
            "pmin": float(np.min(result["p"])),
            "power": compute_power(result["p"], result["D"], result["K"]),
        }

    def solve_hydraulic_from_edge_viscosities(self, mu_edge_mean: np.ndarray) -> Dict[str, Any]:
        """
        Alternativa (Item 5 do PDF): Recebe diretamente a Viscosidade Média (<mu(T)>)
        integrada ao longo da aresta, pulando o cálculo da temperatura média.
        """
        hcfg = self.hydraulic_cfg_from_mu(mu_edge_mean)
        hydraulic_data = hydraulic_conductivities(self.Xno, self.conec, hcfg)
        result = solve_network(self.conec, hydraulic_data["conductance_edge"], hcfg["pressure_bc"], hcfg["flow_bc"], t=0.0)
        return {
            "hydraulic_cfg": hcfg,
            "hydraulic_data": hydraulic_data,
            "result": result,
            "pmax": float(np.max(result["p"])),
            "pmin": float(np.min(result["p"])),
            "power": compute_power(result["p"], result["D"], result["K"]),
        }


# ============================================================
# PLOTS (VISUALIZAÇÃO DE DADOS)
# ============================================================

def save_contour_plot(x: np.ndarray, y: np.ndarray, T: np.ndarray, title: str, filename: str,
                      Xno: Optional[np.ndarray] = None, conec: Optional[np.ndarray] = None) -> None:
    """
    Gera o mapa de calor (isolinhas preenchidas) da placa térmica.
    Se a rede hidráulica for fornecida (Xno, conec), ela é desenhada por cima (sobreposição).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    cf = ax.contourf(x, y, T, levels=24)
    ax.set_aspect("equal") # Garante que 1m em X tenha o mesmo tamanho visual de 1m em Y
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    fig.colorbar(cf, ax=ax, label="T [°C]")

    # === SOBREPOSIÇÃO DA REDE HIDRÁULICA ===
    if Xno is not None and conec is not None:
        segments = [(Xno[i], Xno[j]) for i, j in conec.astype(int)]
        # LineCollection é muito mais rápido que usar plt.plot num loop for para grafos grandes.
        lc = LineCollection(segments, colors="black", linewidths=0.6, alpha=0.4, zorder=2)
        ax.add_collection(lc)
        ax.scatter(Xno[:, 0], Xno[:, 1], color="black", s=3, alpha=0.4, zorder=3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220)
    plt.close(fig) # Libera a memória RAM fechando a figura explicitamente.

def save_graph_nodes_temperature(Xno: np.ndarray, conec: np.ndarray, T_nodes: np.ndarray, title: str, filename: str) -> None:
    """
    Plota apenas a rede (sem a placa), colorindo os NÓS de acordo com a temperatura interpolada.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    segments = [(Xno[i], Xno[j]) for i, j in conec.astype(int)]
    lc = LineCollection(segments, linewidths=0.7, alpha=0.45) # Arestas translúcidas
    ax.add_collection(lc)
    sc = ax.scatter(Xno[:, 0], Xno[:, 1], c=T_nodes, s=22) # Nós coloridos
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label="T nos nós [°C]")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220)
    plt.close(fig)


def save_graph_edges_temperature(Xno: np.ndarray, conec: np.ndarray, T_edge: np.ndarray, title: str, filename: str) -> None:
    """
    Plota apenas a rede, mas colorindo as ARESTAS de acordo com a temperatura média delas.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    segments = [(Xno[i], Xno[j]) for i, j in conec.astype(int)]
    # Mapeia o array T_edge diretamente para as cores do LineCollection
    lc = LineCollection(segments, array=T_edge, linewidths=2.0)
    ax.add_collection(lc)
    ax.scatter(Xno[:, 0], Xno[:, 1], s=4, alpha=0.25) # Nós translúcidos
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    fig.colorbar(lc, ax=ax, label="<T> na aresta [°C]")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220)
    plt.close(fig)


def save_profile_plots(x: np.ndarray, y: np.ndarray, T: np.ndarray, title: str, filename: str) -> None:
    """
    Gera gráficos de linha (Cortes 1D) da temperatura.
    Útil para visualizar os picos causados pelo aquecedor circular ou as perturbações dos microcanais.
    """
    jy = len(y) // 2 # Índice correspondente à linha central horizontal (y = Ly/2)
    # Busca o índice de x mais próximo do centro do círculo
    ix = int(np.argmin(np.abs(x - CONFIG_INTEGRADO["circle_center_x"])))
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    
    # Gráfico 1: Perfil ao longo do eixo X (mantendo Y constante no meio)
    axes[0].plot(x, T[jy, :], marker="o", markersize=2)
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("T [°C]")
    axes[0].set_title("perfil horizontal")
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico 2: Perfil ao longo do eixo Y (mantendo X constante perto do círculo)
    axes[1].plot(y, T[:, ix], marker="o", markersize=2)
    axes[1].set_xlabel("y [m]")
    axes[1].set_ylabel("T [°C]")
    axes[1].set_title("perfil vertical")
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220)
    plt.close(fig)

# ============================================================
# EXERCÍCIOS 4.2.1 DO PDF
# ============================================================

def print_theoretical_4_2_1_item_1() -> None:
    """
    Resposta teórica para o Item 4.2.1 (1).
    
    Fundamento:
    A Regra do Trapézio Composta aproxima a integral assumindo comportamento linear 
    por partes. O erro de truncamento global é da ordem de $O(\Delta s^2)$.
    """
    print("\n" + "=" * 80)
    print("RESPOSTA TEÓRICA - EXERCÍCIO 4.2.1(1)")
    print("=" * 80)
    print("Dedução da Regra do Trapézio Composta para a temperatura média <T_k>:")
    print("\n1. Divisão do Domínio:")
    print("   A aresta de comprimento L_k é dividida em N subintervalos iguais.")
    print("   O passo espacial é Δs = L_k / N.")
    print("\n2. Aplicação do Trapézio Simples:")
    print("   Em cada subintervalo [s_{n-1}, s_n], a área é aproximada por um trapézio:")
    print("   Integral ≈ (Δs/2) * [T(s_{n-1}) + T(s_n)]")
    print("\n3. Composição (Soma dos Subintervalos):")
    print("   Ao somar todos os intervalos, os nós internos (compartilhados entre dois")
    print("   subintervalos vizinhos) são somados duas vezes. As pontas, apenas uma.")
    print("   Integral_Total ≈ (Δs/2) * [T(s_0) + 2*Σ_{n=1}^{N-1} T(s_n) + T(s_N)]")
    print("\n4. Cálculo da Média:")
    print("   Como a média é <T_k> = Integral_Total / L_k, e sabemos que Δs/L_k = 1/N:")
    print("   <T_k> ≈ (1 / 2N) * [T(s_0) + 2*Σ_{n=1}^{N-1} T(s_n) + T(s_N)]")
    print("=" * 80)

def print_theoretical_4_2_1_item_5() -> None:
    """
    Resposta teórica para o Item 4.2.1 (5).
    
    Fundamento Matemático (Desigualdade de Jensen):
    Para qualquer função estritamente convexa (ou côncava) $f(x)$, o valor da função 
    na média dos pontos não é igual à média da função avaliada nos pontos.
    No caso da viscosidade da água, a dependência com a temperatura é altamente não-linear,
    tornando a integração direta da propriedade muito mais precisa fisicamente.
    """
    print("\n" + "=" * 80)
    print("RESPOSTA TEÓRICA - EXERCÍCIO 4.2.1(5)")
    print("=" * 80)
    print("Alternativa para o cálculo da viscosidade efetiva nas arestas:\n")
    print("Em vez de calcular a temperatura média <T> para depois aplicá-la na")
    print("fórmula da viscosidade -- ou seja, usar μ(<T>) --, o mais rigoroso é")
    print("integrar a PRÓPRIA VISCOSIDADE ao longo da aresta para achar <μ>:")
    print("\n   <μ_k> = (1/L_k) * Integral[0 -> L_k] μ(T(p(s))) ds")
    print("\nJustificativa Física e Matemática:")
    print("* A viscosidade da água μ(T) possui um comportamento fortemente NÃO-LINEAR.")
    print("* Em funções não-lineares, a função da média NÃO É IGUAL à média da função")
    print("  (ou seja, μ(<T>) ≠ <μ(T)>).")
    print("* Integrar diretamente a viscosidade ao longo da geometria capta com")
    print("  exatidão a real resistência ao escoamento distribuída pelo canal.")
    print("=" * 80)

def exercise_42_interpolation(model: HydroThermalCoupledModel) -> Dict[str, Dict[str, Any]]:
    """
    Item 4.2.1(2): Resolução da EDP térmica e interpolação em malhas de diferentes resoluções.
    
    Trade-off de Malha:
    A malha "refinada" (241x121) minimiza erros de discretização espacial (truncamento da série de Taylor), 
    mas aumenta cubicamente o tempo de solução do sistema linear. A malha "grosseira" é rápida,
    mas pode não capturar adequadamente o contorno circular e os gradientes térmicos.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for label, mesh, secondary_grid in [
        ("refinada_241x121", model.cfg["thermal_mesh_refined"], model.cfg["secondary_grid_refined_case"]),
        ("grosseira_61x31", model.cfg["thermal_mesh_coarse"], model.cfg["secondary_grid_coarse_case"]),
    ]:
        Nx, Ny = mesh
        print(f"\nEXERCÍCIO 4.2.1(2) - solução térmica base {label}")
        thermal = model.solve_thermal(Nx, Ny)
        results[label] = thermal
        print(f"Tmax = {thermal['Tmax']:.6f} °C | Tmean = {thermal['Tmean']:.6f} °C | tempo = {thermal['total_time']:.3f} s")

        save_contour_plot(thermal["x"], thermal["y"], thermal["T_grid"], f"Campo térmico base - {label}", f"ex-4-2-1_item-2_campo-termico-base_{label}.png", model.Xno, model.conec)

        # Avaliação de interpoladores em grade secundária.
        xs = np.linspace(0.0, model.cfg["Lx"], secondary_grid[0])
        ys = np.linspace(0.0, model.cfg["Ly"], secondary_grid[1])
        XX, YY = np.meshgrid(xs, ys)
        pts = np.column_stack([XX.ravel(), YY.ravel()])

        for method in ["linear", "cubic", "nearest"]:
            interp = TemperatureInterpolator(thermal["x"], thermal["y"], thermal["T_grid"], method=method)
            Tsec = interp(pts).reshape((len(ys), len(xs)))
            save_contour_plot(xs, ys, Tsec, f"Interpolação {method} - {label}", f"ex-4-2-1_item-2_interpolacao-{method}_{label}.png", model.Xno, model.conec)

        # Interpolação para os graus de liberdade da rede (Nós do grafo).
        interp_linear = TemperatureInterpolator(thermal["x"], thermal["y"], thermal["T_grid"], method="linear")
        T_nodes = interp_linear(model.Xno)
        save_graph_nodes_temperature(model.Xno, model.conec, T_nodes, f"Rede colorida por T nodal - {label}", f"ex-4-2-1_item-2_rede-temperatura-nodal_{label}.png")

    return results


def exercise_42_edge_means_and_hydraulics(model: HydroThermalCoupledModel, thermal_results: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Itens 4.2.1(3), 4.2.1(4) e 4.2.1(5): Análise de convergência das regras de quadratura.
    Calcula a redução do erro na integração da temperatura conforme o número de subdivisões ($N$) aumenta.
    """
    quad_rows: List[Dict[str, Any]] = []
    hydro_rows: List[Dict[str, Any]] = []

    print("\nEXERCÍCIO 4.2.1(3) - TEMPERATURA MÉDIA NAS ARESTAS")
    for mesh_label, thermal in thermal_results.items():
        interp = TemperatureInterpolator(thermal["x"], thermal["y"], thermal["T_grid"], method="linear")

        # Solução de referência ("Ground Truth") numérica: Trapézio com altíssimo número de pontos.
        T_ref = mean_edge_temperature(model.Xno, model.conec, interp, "trapezoid", 1000)

        for rule in ["midpoint", "trapezoid"]:
            for N in [1, 10, 100, 1000]:
                t0 = time.perf_counter()
                T_edge = mean_edge_temperature(model.Xno, model.conec, interp, rule, N)
                dt = time.perf_counter() - t0

                # Cálculo de erro via Norma Infinita (Máximo erro absoluto) e Norma L2 (Root Mean Square Error).
                err_inf = float(np.max(np.abs(T_edge - T_ref)))
                err_l2 = float(np.sqrt(np.mean((T_edge - T_ref) ** 2)))

                quad_rows.append({
                    "mesh": mesh_label,
                    "rule": rule,
                    "subdivisions": N,
                    "T_edge_min": float(np.min(T_edge)),
                    "T_edge_mean": float(np.mean(T_edge)),
                    "T_edge_max": float(np.max(T_edge)),
                    "erro_inf_vs_trap1000": err_inf,
                    "erro_l2_vs_trap1000": err_l2,
                    "time_s": dt,
                })

                # Resolve a física do fluido baseada na geometria atualizada.
                hydro = model.solve_hydraulic_from_edge_temperatures(T_edge)
                hydro_rows.append({
                    "mesh": mesh_label,
                    "rule": rule,
                    "subdivisions": N,
                    "pmax_Pa": hydro["pmax"],
                    "pmin_Pa": hydro["pmin"],
                    "power_W": hydro["power"],
                    "mu_edge_mean_Pa_s": float(np.mean(hydro["hydraulic_data"]["mu"])),
                })

                print(f"{mesh_label:18s} | {rule:9s} | N={N:4d} | <T>med={np.mean(T_edge):.4f} °C | erro∞={err_inf:.3e} | tempo={dt:.3f}s | pmax={hydro['pmax']:.3e} Pa | P={hydro['power']:.3e} W")

                if mesh_label.startswith("refinada") and rule == "trapezoid" and N == 100:
                    save_graph_edges_temperature(model.Xno, model.conec, T_edge, "Rede colorida por temperatura média nas arestas", "ex-4-2-1_item-3_rede-temperatura-media-arestas.png")
                    
                    if PlotaRede is not None:
                        p_calc = hydro["result"]["p"]
                        q_calc = hydro["result"]["q"]
                        fig, ax = PlotaRede(model.conec, model.Xno, p_calc, q_calc, factor_units=0.001)
                        ax.set_title(f"Pressão e Fluxo na Rede (Professor) - {mesh_label}")
                        fig.savefig(OUTPUT_DIR / f"ex-4-2-1_item-4_pressao-fluxo-rede_{mesh_label}.png", dpi=300, bbox_inches="tight")
                        plt.close(fig)

        # Alternativa rigorosa (Item 5): integração de viscosidade pura.
        mu_edge_direct = mean_edge_viscosity_direct(model.Xno, model.conec, interp, "trapezoid", 1000)
        hydro_direct = model.solve_hydraulic_from_edge_viscosities(mu_edge_direct)
        hydro_rows.append({
            "mesh": mesh_label,
            "rule": "direct_mu_trapezoid",
            "subdivisions": 1000,
            "pmax_Pa": hydro_direct["pmax"],
            "pmin_Pa": hydro_direct["pmin"],
            "power_W": hydro_direct["power"],
            "mu_edge_mean_Pa_s": float(np.mean(mu_edge_direct)),
        })
        print(f"{mesh_label:18s} | alternativa item 5: <mu(T)> direto | pmax={hydro_direct['pmax']:.3e} Pa | P={hydro_direct['power']:.3e} W")

    quad_df = pd.DataFrame(quad_rows)
    hydro_df = pd.DataFrame(hydro_rows)
    quad_df.to_csv(OUTPUT_DIR / "42_quadraturas_temperatura_media_arestas.csv", index=False)
    hydro_df.to_csv(OUTPUT_DIR / "42_hidraulica_acoplada_resultados.csv", index=False)
    return quad_df, hydro_df


# ============================================================
# EXERCÍCIOS 4.3.3 DO PDF
# ============================================================

def exercise_43_conductivity(model: HydroThermalCoupledModel) -> pd.DataFrame:
    """
    Item 4.3.3(1): Estudo de Sensibilidade - Efeito de d_max no campo k(x,y).
    
    A presença dos canais altera a matriz de coeficientes da EDP térmica.
    Limitação: Modificar k em vez de modelar a convecção explícita é uma técnica
    de homogeneização grosseira, mas atende aos requisitos do enunciado numérico.
    """
    rows: List[Dict[str, Any]] = []
    print("\nEXERCÍCIO 4.3.3(1) - CONDUTIVIDADE MODIFICADA PELA REDE")

    for mesh in model.cfg.get("thermal_meshes_43", [(61, 31), (121, 61)]):
        for dmax in model.cfg["dmax_values"]:
            influence = NetworkInfluenceModel(model.Xno, model.conec, dmax, k0=model.cfg["k_constant"])
            # closure para injetar d_max específico na avaliação de k_func.
            k_func = lambda x, y, infl=influence: infl.k_modified_at_point(x, y)

            t0 = time.perf_counter()
            result = model.solve_thermal(mesh[0], mesh[1], k_function=k_func)
            dt = time.perf_counter() - t0

            rows.append({
                "Nx": mesh[0],
                "Ny": mesh[1],
                "dmax": dmax,
                "Tmax_C": result["Tmax"],
                "Tmean_C": result["Tmean"],
                "time_s": dt,
            })
            print(f"malha={mesh} | dmax={dmax:.5g} | Tmax={result['Tmax']:.5f} °C | Tmean={result['Tmean']:.5f} °C | tempo={dt:.3f}s")

            if mesh in [(61, 31), (241, 121)]:
                suffix = f"{mesh[0]}x{mesh[1]}_dmax_{dmax:g}".replace(".", "p")
                save_contour_plot(result["x"], result["y"], result["T_grid"], f"k modificado - malha {mesh}, dmax={dmax}", f"ex-4-3-3_item-1_campo-termico-k-modificado_{suffix}.png", model.Xno, model.conec)
                save_profile_plots(result["x"], result["y"], result["T_grid"], f"Perfis - k modificado - dmax={dmax}", f"ex-4-3-3_item-1_perfis-temperatura-k-modificado_{suffix}.png")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "43_condutividade_modificada_resultados.csv", index=False)
    return df


def central_spine_intensity(Xno: np.ndarray, conec: np.ndarray) -> np.ndarray:
    """
    Função geométrica auxiliar.
    Identifica o canal central (que liga a entrada principal à saída) analisando
    os vetores direção (delta x > delta y) e as posições em y (~ Ly/2).
    Atribui peso estatístico maior (100.0) à espinha central e 0.1 às ramificações.
    """
    mid = 0.5 * (Xno[conec[:, 0]] + Xno[conec[:, 1]])
    vec = Xno[conec[:, 1]] - Xno[conec[:, 0]]
    y_center = 0.5 * CONFIG_INTEGRADO["Ly"]
    horizontal = np.abs(vec[:, 0]) >= np.abs(vec[:, 1])
    near_center = np.abs(mid[:, 1] - y_center) < 0.00075
    I = np.full(conec.shape[0], 0.1, dtype=float)
    I[horizontal & near_center] = 100.0
    return I


def exercise_43_source_sink(model: HydroThermalCoupledModel) -> pd.DataFrame:
    """
    Item 4.3.3(2): Estudo da magnitude do termo Fonte/Sumidouro modificado (S_p).
    
    Fundamento Matemático (Regularização Estritamente Numérica):
    Fisicamente, a troca de calor ocorre apenas na interface fluido/sólido. Se tentarmos 
    modelar isso numa malha grosseira como um ponto de extração exato, obteremos uma singularidade.
    O espalhamento Gaussiano distribui essa energia por volumes vizinhos na EDP elíptica,
    garantindo que o sistema linear resultante não fique mal condicionado (Condition Number elevado).
    """
    rows: List[Dict[str, Any]] = []
    print("\nEXERCÍCIO 4.3.3(2) - TERMO FONTE/SUMIDOURO DA REDE")

    dmax = 0.0005
    influence = NetworkInfluenceModel(model.Xno, model.conec, dmax, k0=model.cfg["k_constant"])
    intensity_cases = {
        "homogenea_I1": np.ones(model.conec.shape[0], dtype=float),
        "espinha_100_resto_0p1": central_spine_intensity(model.Xno, model.conec),
    }

    for intensity_label, I_edge in intensity_cases.items():
        for S0 in model.cfg["s0_values"]:
            def src_func(x: float, y: float, infl=influence, S0_=S0, I_=I_edge) -> float:
                return model.cfg["source_value"] + infl.source_extra_at_point(x, y, S0_, I_)

            t0 = time.perf_counter()
            result = model.solve_thermal(121, 61, source_function=src_func)
            dt = time.perf_counter() - t0

            rows.append({
                "intensity_case": intensity_label,
                "S0": S0,
                "dmax": dmax,
                "Nx": 121,
                "Ny": 61,
                "Tmax_C": result["Tmax"],
                "Tmean_C": result["Tmean"],
                "time_s": dt,
            })
            print(f"I={intensity_label:22s} | S0={S0: .1e} | Tmax={result['Tmax']:.5f} °C | Tmean={result['Tmean']:.5f} °C | tempo={dt:.3f}s")

            if S0 in (5e5, 1e6):
                suffix = f"{intensity_label}_S0_{S0:.0e}".replace("+", "").replace("-", "menos_").replace(".", "p")
                save_contour_plot(result["x"], result["y"], result["T_grid"], f"Fonte/sumidouro - {intensity_label}, S0={S0:.1e}", f"ex-4-3-3_item-2_campo-termico-fonte-sumidouro_{suffix}.png", model.Xno, model.conec)
                save_profile_plots(result["x"], result["y"], result["T_grid"], f"Perfis - {intensity_label}, S0={S0:.1e}", f"ex-4-3-3_item-2_perfis-temperatura-fonte-sumidouro_{suffix}.png")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "43_fonte_sumidouro_resultados.csv", index=False)
    return df


# ============================================================
# RELATÓRIO TEXTUAL GERADO JUNTO COM OS RESULTADOS
# ============================================================

def write_markdown_report(quad_df: pd.DataFrame, hydro_df: pd.DataFrame,
                          k_df: Optional[pd.DataFrame], source_df: Optional[pd.DataFrame]) -> None:
    """
    Centraliza todas as métricas geradas em um arquivo `.md`.
    
    Decisão de Projeto: 
    Relatórios estruturados e agnósticos ao interpretador garantem que resultados
    estejam legíveis mesmo se o ambiente Python original se perder. A conversão de
    DataFrames Pandas para Markdown é limpa e suportada pelo GitHub/GitLab nativamente.
    """
    lines: List[str] = []
    lines.append("# Relatório de resultados - Acoplamento hidráulico-térmico\n")
    lines.append("## Regra do trapézio composta\n")
    lines.append("Para uma aresta de comprimento L dividida em N subintervalos, Δs=L/N:\n")
    lines.append("`integral ≈ Δs[0.5*T(s0) + Σ T(sn) + 0.5*T(sN)]`.\n")
    lines.append("Logo, a média na aresta é essa integral dividida por L.\n")

    lines.append("## Quadratura de temperatura média nas arestas\n")
    lines.append(quad_df.sort_values(["mesh", "rule", "subdivisions"]).to_markdown(index=False))
    lines.append("\n\n## Rede hidráulica acoplada\n")
    lines.append(hydro_df.sort_values(["mesh", "rule", "subdivisions"]).to_markdown(index=False))

    if k_df is not None:
        lines.append("\n\n## Condutividade térmica modificada pela rede\n")
        lines.append(k_df.to_markdown(index=False))

    if source_df is not None:
        lines.append("\n\n## Fonte/sumidouro da rede de microcanais\n")
        lines.append(source_df.to_markdown(index=False))

    lines.append("\n\n## Resposta conceitual do item 4.2.1(5)\n")
    lines.append("Em vez de calcular a temperatura média da aresta e depois usar μ(<T>), pode-se calcular diretamente a viscosidade média da aresta: <μ> = (1/L)∫μ(T(p(s)))ds. Como μ(T) é não linear, geralmente μ(<T>) não é igual a <μ(T)>.\n")

    (OUTPUT_DIR / "relatorio_resultados_acoplamento.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Ponto de entrada (Entry point) do script.
    Realiza a leitura das "Feature Toggles" na configuração global para determinar
    quais partes computacionalmente intensivas do código devem ser executadas.
    """
    print("=" * 100)
    print("ACOPLAMENTO HIDRÁULICO-TÉRMICO - EXECUÇÃO DOS EXERCÍCIOS DO PDF")
    print("=" * 100)
    print(f"Saídas serão salvas em: {OUTPUT_DIR}")

    model = HydroThermalCoupledModel(CONFIG_INTEGRADO)
    
    if CONFIG_INTEGRADO.get("run_ex_4-2-1_item-1"):
        print_theoretical_4_2_1_item_1()

    # O escopo destas variáveis precisa ser acessível nas linhas subsequentes se
    # as flags estiverem habilitadas.
    thermal_results = {}
    
    if CONFIG_INTEGRADO.get("run_ex_4-2-1_item_2"):
        thermal_results = exercise_42_interpolation(model)
        
    if CONFIG_INTEGRADO.get("run_ex_4-2-1_item_3-4-5"):
        # Tolerância a falhas na configuração: Se o item 2 foi pulado, recalcula a 
        # base térmica para não quebrar a dependência de dados dos itens 3 a 5.
        if not CONFIG_INTEGRADO.get("run_ex_4-2-1_item_2"):
             thermal_results = exercise_42_interpolation(model) 
        quad_df, hydro_df = exercise_42_edge_means_and_hydraulics(model, thermal_results)
        print_theoretical_4_2_1_item_5()

    if CONFIG_INTEGRADO.get("run_ex_4-3-3_item-1"):
        k_df = exercise_43_conductivity(model)

    if CONFIG_INTEGRADO.get("run_ex_4-3-3_item-2"):
        source_df = exercise_43_source_sink(model)

    print("\n" + "=" * 100)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 100)

# O boilerplate padrão idiomático do Python que impede a execução acidental de
# scripts quando eles são apenas importados como bibliotecas por outros arquivos.
if __name__ == "__main__":
    main()