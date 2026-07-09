# -*- coding: utf-8 -*-
"""
membrana-elastica.py

==================================================
MEMBRANA ELÁSTICA DO GÊMEO DIGITAL
==================================================

Objetivos desta versão:
- Modelar computacionalmente o problema de vibração de membranas
  elásticas tensionadas (ex: elastômeros para microfluídica).
- Montar as matrizes de Rigidez (K) e Massa (M) via diferenças finitas 2D.
- Solucionar o problema de autovalores generalizados (KΦ = λMΦ) para 
  extrair frequências naturais e modos de vibração.
- Suportar solvers densos (espectro completo via Método de Francis) e 
  esparsos (modos fundamentais via iteração de subespaços/Lanczos).
- Manter a organização semântica inspirada no módulo térmico.

Dependências:
    pip install numpy matplotlib scipy
"""

from __future__ import annotations

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

# time: medir custo computacional de montagem e solução do problema de autovalores.
import time

# dataclass: simplifica a criação da classe de configuração com parâmetros físicos e numéricos.
from dataclasses import dataclass

# Tipagem: deixa o código mais legível e facilita manutenção futura.
from typing import Any, Dict, List, Optional, Tuple

# numpy: base numérica do código (vetores, malhas bidimensionais e álgebra linear).
import numpy as np

# matplotlib: geração de gráficos, curvas de nível e visualização dos modos de vibração.
import matplotlib.pyplot as plt

# scipy.sparse: criação e manipulação de matrizes esparsas (formato CSR e COO), 
# fundamentais para lidar com malhas 2D refinadas sem estourar a memória.
from scipy import sparse

# eigsh: solver esparso de autovalores para encontrar os primeiros k modos fundamentais.
from scipy.sparse.linalg import eigsh

# eigh: solver denso para encontrar todos os autovalores de matrizes simétricas (Método de Francis).
from scipy.linalg import eigh

# ==================================================
# GUIA DE USO E CONFIGURAÇÃO DO USUÁRIO
# ==================================================

def print_user_guide() -> None:
    """Explica como o usuário pode interagir com o programa."""
    print("=" * 120)
    print("GUIA RÁPIDO DE USO - MEMBRANA ELÁSTICA")
    print("=" * 120)
    print("1) Para mudar o refinamento, altere Nx e Ny em CONFIG.")
    print("2) Para alterar o número de modos calculados, ajuste num_modes.")
    print("3) Use solver_mode='sparse' para malhas finas (mais eficiente).")
    print("4) Ative as flags run_exercise_* para executar os estudos da apostila:")
    print("   - Ex 1: Membrana circular e máscara")
    print("   - Ex 2: Frequências vs Discretização (Tabela)")
    print("   - Ex 3: Dedução teórica de coeficientes")
    print("   - Ex 4: Projeção do termo forçante")
    print("   - Ex 5: Energia média vs Frequência (Ressonância)")
    print("=" * 120)


CONFIG = {
    # ---------------------------
    # Geometria e Discretização
    # ---------------------------
    "radius": 0.004,      # Raio da membrana circular [m] (0.4 cm)
    "Lx_hat": 2.0,        # Dimensão X do domínio adimensional
    "Ly_hat": 2.0,        # Dimensão Y do domínio adimensional
    "Nx": 81,             # Nós na direção X
    "Ny": 81,             # Nós na direção Y
    
    # ---------------------------
    # Propriedades Físicas (Elastômero)
    # ---------------------------
    "thickness": 1.0e-4,  # Espessura [m] (0.1 mm)
    "sigma": 200.0,       # Tensão membranal [N/m]
    "rho": 900.0,         # Densidade [kg/m³]
    "beta_damping": 0.02, # Amortecimento proporcional (beta)

    # ---------------------------
    # Opções de Solução Numérica
    # ---------------------------
    "solver_mode": "sparse", # 'sparse' (eigsh) ou 'dense' (eigh)
    "num_modes": 10,         # Quantidade de modos fundamentais a calcular
    "big_number": 1.0e4,     # Penalização para pontos restritos

    # ---------------------------
    # Saídas e Visualização
    # ---------------------------
    "make_plots": True,      # Gerar gráficos de modos e energia
    "contour_levels": 25,    # Resolução dos contornos coloridos
    
    # ---------------------------
    # Geometria da membrana
    # ---------------------------
    "use_circular_mask": True, # True para círculo, False para quadrado

    # ---------------------------
    # Exercícios: Investigando o Comportamento
    # ---------------------------
    "run_exercise_1": False,  # Caso circular com máscara
    "run_exercise_2": False,  # Tabela de frequências vs malha
    "run_exercise_3": False, # Explicação teórica (c_i e phi_i)
    "run_exercise_4": False,  # Projeção do termo forçante senoidal
    "run_exercise_5": False,  # Gráfico de energia média (loglog)
}

@dataclass
class MembraneConfig:
    """Encapsula as configurações para facilitar o acesso no solver."""
    radius: float
    thickness: float
    sigma: float
    rho: float
    Lx_hat: float
    Ly_hat: float
    Nx: int
    Ny: int
    big_number: float
    solver_mode: str
    num_modes: int
    make_plots: bool
    contour_levels: int
    beta_damping: float
    use_circular_mask: bool

# ============================================================
# FUNÇÕES AUXILIARES DE MAPEAMENTO
# ============================================================

def ij2n(i: int, j: int, Nx: int) -> int:
    """
    Mapeia o par de índices (i, j) da malha 2D para um índice global n (1D).
    
    Fundamento:
    -----------
    Para transformar a equação diferencial parcial em um sistema algébrico, 
    precisamos reordenar os nós da malha em um único vetor contínuo. 
    O nó (i,j) recebe o identificador único n = i + j * Nx.
    """
    return i + j * Nx


def n2ij(n: int, Nx: int) -> Tuple[int, int]:
    """Faz a operação inversa de ij2n: converte o índice global n para (i, j)."""
    j = n // Nx
    i = n % Nx
    return i, j


# ============================================================
# CLASSE BASE E SOLVER PRINCIPAL
# ============================================================

class BaseLinearAlgebraModel:
    """Classe base genérica para armazenar o último resultado calculado."""
    def __init__(self) -> None:
        self.last_result: Optional[Dict[str, Any]] = None


class ElasticMembraneSolver(BaseLinearAlgebraModel):
    """
    Solver principal da membrana elástica.
    
    Responsável por:
    - Construir a malha espacial adimensional.
    - Aplicar a máscara circular (condições de contorno).
    - Montar as matrizes de Rigidez (K) e Massa (M).
    - Resolver o problema generalizado K*Phi = lambda*M*Phi.
    """
    
    def __init__(self, cfg: MembraneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        
        # 1. Criação da malha espacial adimensional (x_hat, y_hat)
        # O domínio computacional vai de -L_hat/2 a L_hat/2
        self.x_hat = np.linspace(-self.cfg.Lx_hat / 2.0, self.cfg.Lx_hat / 2.0, self.cfg.Nx)
        self.y_hat = np.linspace(-self.cfg.Ly_hat / 2.0, self.cfg.Ly_hat / 2.0, self.cfg.Ny)
        
        # Passos espaciais adimensionais
        self.hx_hat = self.cfg.Lx_hat / (self.cfg.Nx - 1)
        self.hy_hat = self.cfg.Ly_hat / (self.cfg.Ny - 1)
        
        # O operador Laplaciano discreto implementado assume malhas regulares (hx = hy)
        if abs(self.hx_hat - self.hy_hat) > 1e-12:
            raise ValueError("Erro: Este código assume hx_hat = hy_hat (malha quadrada).")
        self.h_hat = self.hx_hat
        
        # Número total de incógnitas do sistema linear
        self.nunk = self.cfg.Nx * self.cfg.Ny
        
        # 2. Fator de conversão dimensional (Adimensional -> Hertz)
        # Conforme a formulação do PDF: f_k = (omega_hat / 2*pi) * (1/R) * sqrt(sigma / (rho*e))
        self.freq_factor_hz = (1.0 / (2.0 * np.pi * self.cfg.radius)) * np.sqrt(
            self.cfg.sigma / (self.cfg.rho * self.cfg.thickness)
        )

    # --------------------------------------------------------
    # GEOMETRIA E CONDIÇÕES DE CONTORNO (MÁSCARA)
    # --------------------------------------------------------

    def is_inside_unit_circle(self, x_hat: float, y_hat: float) -> bool:
        """Verifica se o ponto adimensional está dentro do círculo de raio 1."""
        return x_hat * x_hat + y_hat * y_hat <= 1.0

    def is_restricted_point(self, i: int, j: int) -> bool:
        """
        Determina se um nó (i,j) possui movimento restrito.
        """
        # 1. Bordas do domínio quadrado (sempre restritas em ambos os casos)
        if i == 0 or i == self.cfg.Nx - 1 or j == 0 or j == self.cfg.Ny - 1:
            return True

        # Se a máscara circular estiver desligada, a membrana é quadrada
        if not self.cfg.use_circular_mask:
            return False

        # 2. Aplicação da máscara circular [cite: 435, 436]
        xh = self.x_hat[i]
        yh = self.y_hat[j]
        
        if not self.is_inside_unit_circle(xh, yh):
            return True
            
        # Verifica vizinhos para garantir suavidade na borda do círculo
        neighbors = [
            (self.x_hat[i + 1], self.y_hat[j]),
            (self.x_hat[i - 1], self.y_hat[j]),
            (self.x_hat[i], self.y_hat[j + 1]),
            (self.x_hat[i], self.y_hat[j - 1]),
        ]
        for xn, yn in neighbors:
            if not self.is_inside_unit_circle(xn, yn):
                return True
                
        return False

    def build_mask_grid(self) -> np.ndarray:
        """Gera uma matriz booleana 2D para visualização da região ativa da membrana."""
        mask = np.zeros((self.cfg.Ny, self.cfg.Nx), dtype=bool)
        for j in range(self.cfg.Ny):
            for i in range(self.cfg.Nx):
                mask[j, i] = not self.is_restricted_point(i, j)
        return mask

    # --------------------------------------------------------
    # MONTAGEM DO SISTEMA LINEAR (ASSEMBLY)
    # --------------------------------------------------------

    def assembly(self) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """
        Monta a matriz de rigidez (K) e a matriz de massa (M) adimensionais.
        
        Utiliza o formato COO (Coordinate list) internamente pela eficiência
        na inserção de dados, convertendo para CSR (Compressed Sparse Row) no final
        para otimizar as operações algébricas.
        """
        t0 = time.perf_counter()
        
        # Estruturas para K em formato COO
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        
        # M será apenas diagonal nesta formulação
        mass_diag = np.zeros(self.nunk, dtype=float)
        
        stiffness_scale = 1.0 / (self.h_hat ** 2)

        for j in range(self.cfg.Ny):
            for i in range(self.cfg.Nx):
                Ic = ij2n(i, j, self.cfg.Nx)

                # Pontos restritos recebem big_number na diagonal de K
                if self.is_restricted_point(i, j):
                    rows.append(Ic)
                    cols.append(Ic)
                    data.append(self.cfg.big_number)
                    mass_diag[Ic] = 1.0
                    continue

                # Pontos internos: Operador Laplaciano de 5 pontos
                Ie = ij2n(i + 1, j, self.cfg.Nx)
                Iw = ij2n(i - 1, j, self.cfg.Nx)
                In = ij2n(i, j + 1, self.cfg.Nx)
                Is = ij2n(i, j - 1, self.cfg.Nx)

                rows.extend([Ic, Ic, Ic, Ic, Ic])
                cols.extend([Ic, Ie, Iw, In, Is])
                data.extend([
                    4.0 * stiffness_scale,  # Diagonal principal
                    -1.0 * stiffness_scale, # Vizinho Leste
                    -1.0 * stiffness_scale, # Vizinho Oeste
                    -1.0 * stiffness_scale, # Vizinho Norte
                    -1.0 * stiffness_scale, # Vizinho Sul
                ])
                mass_diag[Ic] = 1.0

        # Conversão e criação das matrizes esparsas
        K = sparse.coo_matrix((data, (rows, cols)), shape=(self.nunk, self.nunk)).tocsr()
        M = sparse.diags(mass_diag, offsets=0, format="csr")
        
        t1 = time.perf_counter()
        self.assembly_time = t1 - t0
        
        return K, M

    # --------------------------------------------------------
    # SOLUÇÃO DO SISTEMA (AUTOVALORES E AUTOVETORES)
    # --------------------------------------------------------

    def solve_eigenproblem(
        self, K: sparse.csr_matrix, M: sparse.csr_matrix,
        num_modes: Optional[int] = None, mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resolve o problema de autovalores generalizado K*Phi = lambda*M*Phi.
        Suporta o método denso (útil para extrair todos os modos em malhas pequenas) 
        e o método esparso (útil para extrair apenas os fundamentais em malhas grandes).
        """
        num_modes = num_modes or self.cfg.num_modes
        mode = mode or self.cfg.solver_mode
        t0 = time.perf_counter()

        if mode == "dense":
            Kd = K.toarray()
            Md = M.toarray()
            # Resolve espectro completo
            evals, evecs = eigh(Kd, Md)
            # Ordena e corta para manter apenas os 'num_modes' requisitados
            idx = np.argsort(evals)
            evals = evals[idx][:num_modes]
            evecs = evecs[:, idx][:, :num_modes]
        elif mode == "sparse":
            # Extrai os menores autovalores em magnitude ('SM') de forma iterativa
            evals, evecs = eigsh(K, k=num_modes, M=M, which="SM")
            idx = np.argsort(evals)
            evals = evals[idx]
            evecs = evecs[:, idx]
        else:
            raise ValueError("eig_solver_mode deve ser 'dense' ou 'sparse'.")

        t1 = time.perf_counter()
        
        # Filtra eventuais resíduos numéricos negativos antes da raiz quadrada
        omega_hat = np.sqrt(np.maximum(evals, 0.0))
        freq_hz = omega_hat * self.freq_factor_hz
        
        result = {
            "K": K, 
            "M": M, 
            "evals": evals, 
            "evecs": evecs, 
            "omega_hat": omega_hat, 
            "freq_hz": freq_hz,
            "eigen_time": t1 - t0, 
            "assembly_time": getattr(self, "assembly_time", np.nan),
            "total_time": getattr(self, "assembly_time", 0.0) + (t1 - t0),
        }
        self.last_result = result
        return result

    def solve_system(self) -> Dict[str, Any]:
        """Agrupa a montagem das matrizes e a solução em uma única chamada conveniente."""
        K, M = self.assembly()
        return self.solve_eigenproblem(K, M)

    # --------------------------------------------------------
    # UTILITÁRIOS PARA OS EXERCÍCIOS DA APOSTILA
    # --------------------------------------------------------

    def mode_vector_to_grid(self, mode_vector: np.ndarray) -> np.ndarray:
        """Redimensiona um autovetor 1D de volta para o formato de malha 2D (Ny x Nx)."""
        return mode_vector.reshape((self.cfg.Ny, self.cfg.Nx))

    def get_force_vector_for_exercise_4(self) -> np.ndarray:
        """
        Monta o vetor de força Z baseado na equação do termo forçante (PDF - Ex. 4).
        A força é aplicada apenas na região interna da membrana.
        """
        Z = np.zeros(self.nunk, dtype=float)
        for j in range(self.cfg.Ny):
            for i in range(self.cfg.Nx):
                n = ij2n(i, j, self.cfg.Nx)
                if self.is_restricted_point(i, j):
                    Z[n] = 0.0
                else:
                    xh = self.x_hat[i]
                    yh = self.y_hat[j]
                    Z[n] = (xh - 0.5) ** 2 + (yh - 0.5) ** 2
        return Z

    def generalized_modal_projection(self, M: sparse.csr_matrix, evecs: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Projeta o termo forçante Z na base de autovetores M-ortonormais.
        Retorna o vetor de coeficientes alfa.
        """
        return evecs.T @ Z

    def mean_elastic_energy_curve(self, omega_star_values: np.ndarray, omega_hat: np.ndarray,
                                  alphas: np.ndarray, beta: float = 0.02) -> np.ndarray:
        """
        Calcula a curva de energia elástica média ao longo de um espectro de frequências.
        Isso evidenciará os picos de ressonância onde w* se aproxima de w_hat.
        """
        energies = np.zeros_like(omega_star_values, dtype=float)
        for m, omega_star in enumerate(omega_star_values):
            denom = np.sqrt((omega_hat ** 2 - omega_star ** 2) ** 2 + (beta ** 2) * (omega_star ** 2))
            ci = alphas / denom
            energies[m] = 0.25 * np.sum((ci ** 2) * (omega_hat ** 2))
        return energies
    
    # --------------------------------------------------------
    # IMPRESSÕES E RELATÓRIOS DE TERMINAL
    # --------------------------------------------------------

    def print_inputs_summary(self) -> None:
        """Imprime o resumo das entradas do problema de forma detalhada."""
        print("=" * 110)
        print("ENTRADAS UTILIZADAS NO PROBLEMA DA MEMBRANA ELÁSTICA")
        print("=" * 110)
        print(f"Raio (radius) [m]                   : {self.cfg.radius}")
        print(f"Espessura (thickness) [m]           : {self.cfg.thickness}")
        print(f"Tensão (sigma) [N/m]                : {self.cfg.sigma}")
        print(f"Densidade (rho) [kg/m³]             : {self.cfg.rho}")
        print(f"Nx                                  : {self.cfg.Nx}")
        print(f"Ny                                  : {self.cfg.Ny}")
        print(f"Número total de incógnitas (nunk)   : {self.nunk}")
        print(f"Passo adimensional (h_hat)          : {self.h_hat:.6e}")
        print(f"Solver de autovalores               : {self.cfg.solver_mode}")
        print(f"Modos fundamentais calculados       : {self.cfg.num_modes}")
        print(f"Penalidade de fronteira (big_number): {self.cfg.big_number:.1e}")
        print("=" * 110)

    def print_matrix_info(self, K: sparse.csr_matrix, M: sparse.csr_matrix) -> None:
        """Imprime informações estruturais das matrizes de Rigidez e Massa."""
        print("=" * 110)
        print("INFORMAÇÕES MATRICIAIS")
        print("=" * 110)
        print(f"Dimensão do sistema (N x N)         : {K.shape[0]} x {K.shape[1]}")
        print(f"Não-nulos em K (nnz)                : {K.nnz}")
        print(f"Não-nulos em M (nnz)                : {M.nnz}")
        print(f"Densidade de K                      : {K.nnz / (K.shape[0] * K.shape[1]):.6e}")
        print(f"Densidade de M                      : {M.nnz / (M.shape[0] * M.shape[1]):.6e}")
        print("=" * 110)

    def print_output_summary(self, result: Dict[str, Any]) -> None:
        """Imprime as saídas de tempo e as frequências naturais encontradas."""
        print("=" * 110)
        print("RESULTADOS PRINCIPAIS")
        print("=" * 110)
        print(f"Tempo de montagem (Assembly) [s]    : {result['assembly_time']:.6e}")
        print(f"Tempo de solução (Eigen) [s]        : {result['eigen_time']:.6e}")
        print(f"Tempo total [s]                     : {result['total_time']:.6e}")
        print()
        print(f"Primeiras {len(result['freq_hz'])} frequências naturais:")
        for k, f in enumerate(result["freq_hz"], start=1):
            print(f"Modo {k:2d} -> f = {f:.6f} Hz")
        print("=" * 110)

    def print_final_explanation(self, result: Dict[str, Any]) -> None:
        """Explica ao usuário o fluxo de operações realizadas pelo código."""
        print("=" * 110)
        print("COMENTÁRIO FINAL AO USUÁRIO")
        print("=" * 110)
        print("O programa acabou de:")
        print("1. Discretizar a região da membrana em uma malha 2D (diferenças finitas).")
        print("2. Construir uma máscara circular para restringir os pontos fora do domínio.")
        print("3. Montar a matriz de rigidez K (Laplaciano) e a matriz de massa M.")
        print("4. Resolver o problema generalizado K*Phi = lambda*M*Phi no domínio adimensional.")
        print("5. Converter os autovalores adimensionais em frequências naturais (Hz).")
        print("6. Organizar os autovetores para visualização dos modos de vibração.")
        print("=" * 110)
    
    # --------------------------------------------------------
    # PLOTAGEM E VISUALIZAÇÃO (MATPLOTLIB)
    # --------------------------------------------------------

    def plot_mask(self, filename: str = "0_fig-1_mascara.png") -> None:
        """Plota a máscara booleana para verificar visualmente o domínio circular da membrana."""
        mask = self.build_mask_grid()
        X, Y = np.meshgrid(self.x_hat, self.y_hat)
        plt.figure(figsize=(5.5, 5))
        plt.contourf(X, Y, mask.astype(float), levels=1)
        plt.xlabel("x̂")
        plt.ylabel("ŷ")
        plt.title("Máscara da membrana circular")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_mode(self, mode_vector: np.ndarray, title: str = "", filename: str = "0_fig-2_modo.png") -> None:
        """Renderiza um único modo de vibração em um gráfico de contorno 2D."""
        X, Y = np.meshgrid(self.x_hat, self.y_hat)
        W = self.mode_vector_to_grid(mode_vector)
        plt.figure(figsize=(6.5, 5.2))
        contour = plt.contourf(X, Y, W, levels=self.cfg.contour_levels)
        plt.colorbar(contour, label="Amplitude modal")
        plt.xlabel("x̂")
        plt.ylabel("ŷ")
        plt.title(title if title else "Modo de vibração")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_first_modes(self, result: Dict[str, Any], how_many: Optional[int] = None, ex_prefix: str = "0", start_fig: int = 2) -> None:
        """Plota em sequência os primeiros modos de vibração calculados com numeração contínua."""
        if how_many is None:
            how_many = min(10, result["evecs"].shape[1])
        for k in range(how_many):
            title = f"Modo {k+1} - f = {result['freq_hz'][k]:.6f} Hz"
            # Formata: [ex_prefix]_fig-[start_fig + k]_frequencia-[k+1].png
            filename = f"{ex_prefix}_fig-{start_fig + k}_frequencia-{k+1}.png"
            self.plot_mode(result["evecs"][:, k], title=title, filename=filename)

    def plot_frequency_table_curve(self, table_data: Dict[str, Any], filename: str = "2_fig-1_curva-frequencias.png") -> None:
        """Gera o gráfico de convergência das frequências em função do refinamento da malha."""
        grids = table_data["grid_labels"]
        freqs = table_data["frequencies_matrix"]
        plt.figure(figsize=(10, 5))
        for mode in range(freqs.shape[1]):
            plt.plot(grids, freqs[:, mode], marker="o", label=f"Modo {mode+1}")
        plt.xlabel("Discretização")
        plt.ylabel("Frequência [Hz]")
        plt.title("Frequências naturais vs discretização")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_mean_energy_curve(self, omega_star_values: np.ndarray, energies: np.ndarray, filename: str = "5_fig-1_energia-media.png") -> None:
        """Plota a energia elástica média mostrando os picos de ressonância."""
        plt.figure(figsize=(9, 4.8))
        plt.loglog(omega_star_values, energies)
        plt.xlabel("ω* adimensional")
        plt.ylabel("Energia elástica média")
        plt.title("Energia elástica média vs frequência de excitação")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

# ============================================================
# CLASSE DE EXERCÍCIOS (INVESTIGANDO O COMPORTAMENTO)
# ============================================================

class MembraneExercises:
    """
    Classe que organiza e executa os exercícios propostos na apostila.
    Cada método corresponde a uma etapa do estudo da membrana elástica.
    """

    def __init__(self, base_cfg: MembraneConfig) -> None:
        self.base_cfg = base_cfg

    def clone_cfg(self, **kwargs: Any) -> MembraneConfig:
        """Cria uma cópia da configuração base com parâmetros alterados."""
        data = self.base_cfg.__dict__.copy()
        data.update(kwargs)
        return MembraneConfig(**data)

    def exercise_1_circular_membrane(self) -> Dict[str, Any]:
        """
        Executa o Exercício 1: Modelagem de uma membrana circular.
        Este método força o uso da máscara circular e valida a malha, 
        as condições de contorno e o cálculo das frequências iniciais.
        """
        print("\n" + "#" * 110)
        print("EXERCÍCIO 1 - MEMBRANA CIRCULAR COM MÁSCARA")
        print("#" * 110)
        
        # Cria uma configuração específica para este exercício.
        # Força 'use_circular_mask=True' para garantir a geometria circular, 
        # independente do que estiver definido no dicionário CONFIG global.
        cfg_circular = self.clone_cfg(use_circular_mask=True)
        
        # Inicializa o solver com a configuração circular
        solver = ElasticMembraneSolver(cfg_circular)
        
        # 1. Relatório de Entradas: Exibe os parâmetros físicos e numéricos no terminal
        solver.print_inputs_summary()
        
        # 2. Geração da Máscara: Salva a imagem da geometria caso os plots estejam ativos
        if cfg_circular.make_plots:
            # Salva o gráfico seguindo a nomenclatura: [Exercicio]_fig-[Numero]_[Descricao]
            solver.plot_mask(filename="1_fig-1_mascara.png")
            
        # 3. Solução Numérica: Monta as matrizes K e M e resolve o problema de autovalores
        result = solver.solve_system()
        
        # 4. Informações Técnicas: Exibe a densidade e dimensões das matrizes K e M
        solver.print_matrix_info(result["K"], result["M"])
        
        # 5. Resultados Principais: Lista as primeiras frequências naturais convertidas para Hz
        solver.print_output_summary(result)
        
        # 6. Explicação Final: Resumo didático dos passos executados pelo código
        solver.print_final_explanation(result)
        
        return result

    def exercise_2_frequencies_and_modes(self) -> Dict[str, Any]:
        """
        Executa o Exercício 2: Frequências naturais vs Discretização.
        Calcula as frequências para malhas de diferentes tamanhos, plota os 
        modos para a malha base e gera o gráfico de convergência[cite: 437, 438, 439, 440].
        Respeita a configuração de geometria (circular ou quadrada) do CONFIG.
        """
        forma = "CIRCULAR" if self.base_cfg.use_circular_mask else "QUADRADA"
        print("\n" + "#" * 110)
        print(f"EXERCÍCIO 2 - FREQUÊNCIAS VS DISCRETIZAÇÃO ({forma})")
        print("#" * 110)
        
        # Lista de refinamentos sugerida pela apostila [cite: 439]
        grid_list = [(21, 21), (41, 41), (61, 61), (81, 81), (101, 101)]
        grid_labels = []
        frequencies_rows = []

        for Nx, Ny in grid_list:
            # Clona a configuração alterando apenas a malha da iteração atual
            cfg = self.clone_cfg(Nx=Nx, Ny=Ny)
            solver = ElasticMembraneSolver(cfg)
            result = solver.solve_system()
            
            grid_labels.append(f"{Nx}x{Ny}")
            frequencies_rows.append(result["freq_hz"])
            print(f"Resolvido para malha {Nx}x{Ny}...")

            # Plota os modos de vibração APENAS para a malha base escolhida no CONFIG
            if cfg.make_plots and (Nx, Ny) == (self.base_cfg.Nx, self.base_cfg.Ny):
                # Vai gerar imagens: 2_fig-1_frequencia-1.png até 2_fig-10_frequencia-10.png
                solver.plot_first_modes(result, ex_prefix="2", start_fig=1)

        # Empilha os resultados para formatar a tabela
        frequencies_matrix = np.vstack(frequencies_rows)
        
        # Exibição da tabela de resultados de convergência [cite: 440]
        print("\n" + "=" * 110)
        print("TABELA DE FREQUÊNCIAS NATURAIS [Hz]")
        print("=" * 110)
        header = "Malha      " + "".join([f"| Modo {k+1:2d}      " for k in range(self.base_cfg.num_modes)])
        print(header)
        for label, row in zip(grid_labels, frequencies_matrix):
            line = f"{label:10s}" + "".join([f"| {val:12.4f} " for val in row])
            print(line)
        print("=" * 110)

        # Plota a curva de convergência se os gráficos estiverem ativados
        if self.base_cfg.make_plots:
            # Calcula o próximo número de figura para não sobrescrever os modos
            num_modos_plotados = min(10, self.base_cfg.num_modes)
            prox_fig = num_modos_plotados + 1
            
            # Gera imagem: Ex: 2_fig-11_curva-frequencias.png
            nome_arquivo = f"2_fig-{prox_fig}_curva-frequencias.png"
            ElasticMembraneSolver(self.base_cfg).plot_frequency_table_curve({
                "grid_labels": grid_labels, "frequencies_matrix": frequencies_matrix
            }, filename=nome_arquivo)
            
        return {"grid_labels": grid_labels, "frequencies_matrix": frequencies_matrix}

    def exercise_3_theoretical_coefficients(self) -> Dict[str, Any]:
        """Exercício 3: Detalhamento teórico das fórmulas analíticas e cálculo prático."""
        print("\n" + "#" * 110)
        print("EXERCÍCIO 3 - EXPLICAÇÃO TEÓRICA E CÁLCULO DOS COEFICIENTES")
        print("#" * 110)
        print("Para oscilações livres, a solução é dada pela superposição modal:")
        print("    w(t) = Σ c_k * Φ^(k) * sin(ω_k * t + φ_k)")
        print("\nConsiderando as condições iniciais w(0) = U e w'(0) = V:")
        print("1. Projeta-se U e V na base modal: u_k = Φ_kᵀ * M * U  e  v_k = Φ_kᵀ * M * V")
        print("2. Das relações trigonométricas: c_k * sin(φ_k) = u_k  e  c_k * ω_k * cos(φ_k) = v_k")
        print("3. Obtém-se:")
        print("    c_k = sqrt( u_k² + (v_k / ω_k)² )")
        print("    φ_k = arctan2( u_k, v_k / ω_k )")
        print("-" * 110)
        
        # Inicia o solver para obter matrizes e modos
        solver = ElasticMembraneSolver(self.base_cfg)
        result = solver.solve_system()
        
        # Definição de condições iniciais arbitrárias
        # Deslocamento inicial U: Formato parabólico do Ex 4 (maior no centro, 0 nas bordas)
        U = solver.get_force_vector_for_exercise_4() 
        # Velocidade inicial V: Repouso absoluto
        V = np.zeros_like(U)
        
        # Projeção ortogonal na base de autovetores
        u = solver.generalized_modal_projection(result["M"], result["evecs"], U)
        v = solver.generalized_modal_projection(result["M"], result["evecs"], V)
        
        omega = result["omega_hat"]
        c = np.zeros_like(omega)
        phi = np.zeros_like(omega)
        
        # Cálculo vetorial dos coeficientes c_k e phi_k
        for k in range(len(omega)):
            if omega[k] > 1e-12: # Evita divisão por zero
                c[k] = np.sqrt(u[k]**2 + (v[k]/omega[k])**2)
                phi[k] = np.arctan2(u[k], v[k]/omega[k])
            else:
                c[k] = 0.0
                phi[k] = 0.0
                
        print("\nCálculo computacional das condições iniciais:")
        print("  U (Deslocamento) = Perfil parabólico.")
        print("  V (Velocidade)   = 0.0 (Repouso).")
        print("\nResultados para os modos fundamentais calculados:")
        print("Modo |        ω_k (adimensional) |                c_k |           φ_k (rad) |")
        print("-" * 85)
        for k in range(min(10, len(omega))):
            print(f"{k+1:4d} | {omega[k]:25.6f} | {c[k]:18.6e} | {phi[k]:19.6f} |")
        print("#" * 110)
        
        return {"c": c, "phi": phi, "U": U, "V": V}

    def exercise_4_forcing_projection(self) -> Dict[str, Any]:
        """
        Executa o Exercício 4: Projeção do termo forçante espacial na base modal[cite: 442].
        Calcula os coeficientes α_i que representam a distribuição da força 
        nas coordenadas naturais (modos) da membrana[cite: 444].
        Respeita a geometria (circular ou quadrada) definida no CONFIG.
        """
        forma = "CIRCULAR" if self.base_cfg.use_circular_mask else "QUADRADA"
        print("\n" + "#" * 110)
        print(f"EXERCÍCIO 4 - PROJEÇÃO DO TERMO FORÇANTE ({forma})")
        print("#" * 110)
        
        # Inicializa o solver com a configuração base
        solver = ElasticMembraneSolver(self.base_cfg)
        
        # 1. Solução do problema de autovalores para obter a base modal Φ
        result = solver.solve_system()
        
        # 2. Construção do vetor espacial da força Z [cite: 443, 461]
        # Componentes: Z_ij = (x_hat_ij - 0.5)^2 + (y_hat_ij - 0.5)^2
        Z = solver.get_force_vector_for_exercise_4()
        
        # 3. Projeção Modal: Cálculo dos coeficientes α_i tal que Z = Σ α_i * M * Φ_i [cite: 399]
        # Pela M-ortonormalidade dos modos, α_i = Φ_iᵀ * Z
        alphas = solver.generalized_modal_projection(result["M"], result["evecs"], Z)
        
        print("\nRepresentação do termo forçante na base modal:")
        print(f"Equação: Z = Σ α_i * M * Φ_i")
        print("-" * 60)
        print(f"{'Modo (i)':<12} | {'Coeficiente de Projeção α_i':<25}")
        print("-" * 60)
        # Exibe os 10 primeiros coeficientes para análise detalhada no terminal
        for i, alpha in enumerate(alphas[:10], start=1):
            print(f"{i:<12d} | {alpha:25.8e}")
        print("-" * 60)
            
        # 4. Visualização da distribuição espacial da força aplicada
        if self.base_cfg.make_plots:
            X, Y = np.meshgrid(solver.x_hat, solver.y_hat)
            Zgrid = Z.reshape((solver.cfg.Ny, solver.cfg.Nx))
            
            plt.figure(figsize=(6.5, 5))
            contour = plt.contourf(X, Y, Zgrid, levels=solver.cfg.contour_levels)
            plt.colorbar(contour, label="Intensidade do Forçamento Z(x̂,ŷ)")
            plt.xlabel("x̂")
            plt.ylabel("ŷ")
            plt.title(f"Distribuição Espacial da Força - Membrana {forma.capitalize()}")
            plt.axis("equal")
            plt.tight_layout()
            
            # Nomenclatura obrigatória: 4_fig-1_termo-forcante.png
            plt.savefig("4_fig-1_termo-forcante.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        print("\nCálculo concluído. Distribuição espacial salva em '4_fig-1_termo-forcante.png'.")
        print("#" * 110)
            
        return {"result": result, "alphas": alphas, "Z": Z}

    def exercise_5_mean_elastic_energy(self) -> Dict[str, Any]:
        """
        Executa o Exercício 5: Energia elástica média (Ressonância).
        Calcula a curva de energia para diferentes valores de amortecimento (beta)
        ao longo de um espectro de frequências de excitação (ω*).
        Respeita a geometria (circular ou quadrada) definida no CONFIG.
        """
        forma = "CIRCULAR" if self.base_cfg.use_circular_mask else "QUADRADA"
        print("\n" + "#" * 110)
        print(f"EXERCÍCIO 5 - ENERGIA ELÁSTICA MÉDIA ({forma})")
        print("#" * 110)
        
        # Inicializa o solver com a configuração base
        solver = ElasticMembraneSolver(self.base_cfg)
        
        # 1. Solução do problema de autovalores
        result = solver.solve_system()
        
        # 2. Obtém os coeficientes da projeção modal da força (do Ex 4)
        Z = solver.get_force_vector_for_exercise_4()
        alphas = solver.generalized_modal_projection(result["M"], result["evecs"], Z)
        
        # 3. Definição do espectro de frequências de excitação [0.5, 100]
        # Usamos logspace para garantir boa resolução nos picos de ressonância (escala log)
        omega_star_values = np.logspace(np.log10(0.5), np.log10(100), 500)
        betas = [0.01, 0.1, 1.0] # Valores de amortecimento sugeridos na apostila
        
        print(f"Calculando energias médias para os amortecimentos (β): {betas}")
        energies_dict = {}
        
        # 4. Cálculo e geração do gráfico de ressonância
        if self.base_cfg.make_plots:
            plt.figure(figsize=(10, 6))
            
            for b in betas:
                # Calcula a energia usando a superposição modal
                energies = solver.mean_elastic_energy_curve(
                    omega_star_values, result["omega_hat"], alphas, beta=b
                )
                energies_dict[f"beta_{b}"] = energies
                
                # Plotagem em escala log-log conforme exigido
                plt.loglog(omega_star_values, energies, label=f"β = {b}")
                
            plt.xlabel("ω* (Frequência de excitação)")
            plt.ylabel("Energia Elástica Média")
            plt.title(f"Resposta em Frequência (Ressonância) - Membrana {forma.capitalize()}")
            plt.grid(True, which="both", ls="-", alpha=0.3)
            plt.legend()
            
            # Nomenclatura obrigatória: 5_fig-1_energia-media.png
            plt.savefig("5_fig-1_energia-media.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("\nCálculo concluído. Gráfico de ressonância salvo em '5_fig-1_energia-media.png'.")
        else:
            # Apenas calcula os dados se os plots estiverem desativados
            for b in betas:
                energies = solver.mean_elastic_energy_curve(
                    omega_star_values, result["omega_hat"], alphas, beta=b
                )
                energies_dict[f"beta_{b}"] = energies
            print("\nCálculo concluído (Geração de gráficos desativada).")
            
        print("#" * 110)
            
        return {"omega_star_values": omega_star_values, "energies_dict": energies_dict}

# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main() -> None:
    print_user_guide()
    
    # 1. Instanciação da configuração base a partir do dicionário global
    base_cfg = MembraneConfig(
        radius=CONFIG["radius"],
        thickness=CONFIG["thickness"],
        sigma=CONFIG["sigma"],
        rho=CONFIG["rho"],
        Lx_hat=CONFIG["Lx_hat"],
        Ly_hat=CONFIG["Ly_hat"],
        Nx=CONFIG["Nx"],
        Ny=CONFIG["Ny"],
        big_number=CONFIG["big_number"],
        solver_mode=CONFIG["solver_mode"],
        num_modes=CONFIG["num_modes"],
        make_plots=CONFIG["make_plots"],
        contour_levels=CONFIG["contour_levels"],
        beta_damping=CONFIG["beta_damping"],
        use_circular_mask=CONFIG["use_circular_mask"],
    )

    exercises = MembraneExercises(base_cfg)

    # 2. Execução seletiva dos exercícios
    run_any = False
    
    if CONFIG.get("run_exercise_1", False):
        exercises.exercise_1_circular_membrane()
        run_any = True
        
    if CONFIG.get("run_exercise_2", False):
        exercises.exercise_2_frequencies_and_modes()
        run_any = True
        
    if CONFIG.get("run_exercise_3", False):
        exercises.exercise_3_theoretical_coefficients()
        run_any = True
        
    if CONFIG.get("run_exercise_4", False):
        exercises.exercise_4_forcing_projection()
        run_any = True
        
    if CONFIG.get("run_exercise_5", False):
        exercises.exercise_5_mean_elastic_energy()
        run_any = True

    # 3. Execução padrão caso nenhum exercício específico tenha sido selecionado
    if not run_any:
        print("\n" + "#" * 88)
        print("CASO ÚNICO - RESOLUÇÃO DA MEMBRANA BASE")
        print("#" * 88)
        
        solver = ElasticMembraneSolver(base_cfg)
        solver.print_inputs_summary()
        result = solver.solve_system()
        
        solver.print_output_summary(result)
        solver.print_matrix_info(result["K"], result["M"])
        solver.print_final_explanation(result)
        
        if base_cfg.make_plots:
            solver.plot_mask(filename="0_fig-1_mascara.png")
            solver.plot_first_modes(result, ex_prefix="0", start_fig=2)


if __name__ == "__main__":
    main()
