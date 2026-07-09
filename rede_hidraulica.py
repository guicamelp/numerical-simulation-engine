r"""
===============================================================================
MÓDULO 1: REDE HIDRÁULICA DO GÊMEO DIGITAL
===============================================================================

1. INTUIÇÃO E FÍSICA DO SISTEMA
Este módulo simula o comportamento de fluidos em regime laminar dentro de uma 
rede de microcanais. O sistema físico é abstraído como um grafo matemático onde:
- Nós (vértices) representam uniões e armazenam a variável de pressão ($p$).
- Arestas representam os microcanais e conduzem a variável de vazão ($Q$).

2. FUNDAMENTOS MATEMÁTICOS E MODELAGEM
A modelagem realiza a transição de um sistema contínuo para um discreto.
A base física repousa em dois princípios:
- Conservação de Massa Nodal: A soma algébrica das vazões em qualquer nó i 
  deve ser nula:
  $$\sum Q_{k}^{(i)} = 0$$
- Lei Constitutiva (Hagen-Poiseuille): A vazão em um microcanal k é 
  diretamente proporcional à queda de pressão entre seus extremos:
  $$Q_{k} = C_{k}(p_{i}^{k} - p_{j}^{k})$$
  onde $C_k$ é a condutância hidráulica, dependente da geometria do canal e da 
  viscosidade do fluido $\mu$.

A aplicação da conservação de massa em todos os nós da topologia gera um 
sistema algébrico linear da forma:
  $$A p = b$$
A matriz global de condutância $A$ é simétrica e singular. Para 
garantir a unicidade da solução, o sistema exige a quebra dessa singularidade 
através da imposição de Condições de Contorno (Pressões de Dirichlet ou Vazões 
de Neumann).

3. ARQUITETURA DE CÓDIGO E DECISÕES DE PROJETO
O projeto adota o paradigma Orientado a Objetos (POO). O estado do 
grafo e das matrizes é encapsulado na classe `RedeHidraulica`.

Para garantir eficiência e escalabilidade, as seguintes diretrizes numéricas 
foram implementadas:
- Matrizes Esparsas: Emprego de scipy.sparse (lil_matrix para montagem, 
  csr_matrix para resolução) devido à alta esparsidade do grafo, evitando 
  estouro de RAM.
- Vetorização: O processo de Assembly da matriz $A$ utiliza indexação 
  avançada do NumPy, eliminando laços iterativos lentos.
- Contornos Generalistas: Capacidade de receber dicionários mapeando múltiplos 
  nós para múltiplos valores de contorno simultâneos.
===============================================================================
"""

# =============================================================================
# DEPENDÊNCIAS E TIPAGEM
# =============================================================================

import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

# Módulos para Matrizes Esparsas (Obrigatório para O(n) de memória)
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve

# Manipulação Geométrica e Topológica
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

# Visualização
import matplotlib
# Configuração de backend para evitar falhas de renderização no WSL
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib import cm

# =============================================================================
# PARTE 3: CONFIGURAÇÃO E ENCAPSULAMENTO DO ESTADO (POO)
# =============================================================================

class RedeHidraulica:
    """
    Classe responsável por encapsular a topologia, os parâmetros físicos,
    as condições de contorno e o estado algébrico da rede hidráulica.
    """
    def __init__(self, configuracao: dict):
        """
        Inicializa o subsistema hidráulico definindo o estado interno.
        
        Decisão de projeto: O uso de um dicionário de configuração injetado no 
        construtor isola os parâmetros de entrada. Os dicionários de contorno 
        permitem mapeamentos generalistas de múltiplos nós simultaneamente.
        """
        # 1. Parâmetros de Malha e Escala
        self.levels: int = configuracao.get("levels", 3)
        self.coord_scale_to_m: float = configuracao.get("coord_scale_to_m", 1e-3)
        
        # 2. Parâmetros do Modelo Temporal
        self.time_mode: str = configuracao.get("time_mode", "transient")
        self.t0: float = configuracao.get("t0", 0.0)
        self.tf: float = configuracao.get("tf", 10.0)
        self.dt: float = configuracao.get("dt", 0.2)
        
        # 3. Propriedades Geométricas dos Canais
        self.geometry_mode: str = configuracao.get("geometry_mode", "rectangular")
        self.area_constant: float = configuracao.get("area_constant", 2.5e-7)
        self.diameter_constant: float = configuracao.get("diameter_constant", 800e-6)
        self.width: float = configuracao.get("width", 500e-6)
        self.height: float = configuracao.get("height", 500e-6)
        self.area_per_edge: Optional[np.ndarray] = configuracao.get("area_per_edge", None)
        
        # 4. Condições de Contorno Gerais (Mapeamentos por Dicionário)
        self.pressure_bc: Dict[int, float] = configuracao.get("pressure_bc", {})
        self.flow_bc: Dict[int, dict] = configuracao.get("flow_bc", {})
        
        # 5. Propriedades Térmicas e de Viscosidade do Fluido
        self.temperature_celsius: float = configuracao.get("temperature_celsius", 25.0)
        self.mu_override: Optional[float] = configuracao.get("mu_override", None)
        
        # 6. Estado Topológico da Rede (Matrizes de Coordenadas e Conectividade)
        self.Xno: Optional[np.ndarray] = None     # Coordenadas espaciais dos nós (nv x 2)
        self.conec: Optional[np.ndarray] = None   # Conectividade do grafo (nc x 2)
        self.nv: int = 0                          # Número total de nós (vértices)
        self.nc: int = 0                          # Número total de canais (arestas)
        
        # 7. Estado Algébrico Esparso (Matrizes e Vetores do Sistema)
        self.A_global: Optional[lil_matrix] = None   # Matriz de condutância (LIL para montagem)
        self.b_global: Optional[np.ndarray] = None   # Vetor de forças (vazões nodais)
        self.p: Optional[np.ndarray] = None          # Vetor de pressões nodais calculadas
        self.q: Optional[np.ndarray] = None          # Vetor de vazões nas arestas calculadas

    # =============================================================================
    # PARTE 4: MÉTODOS GERAIS DA REDE (TOPOLOGIA, GEOMETRIA E ASSEMBLY)
    # =============================================================================

    def gerar_topologia_grafo(self) -> None:
        """
        Gera a estrutura base do grafo fractal e os coletores (manifolds),
        armazenando os arrays de nós e conexões diretamente no estado do objeto.
        """
        nodes_data = []
        edges_raw = []
        node_id = 0
        spine_length = 6

        # 1. Geração da Espinha Dorsal Base
        spine_nodes = []
        for i in range(spine_length):
            nodes_data.append({'x': i * 4.0, 'y': 0.0})
            spine_nodes.append(node_id)
            if i > 0: 
                edges_raw.append((node_id - 1, node_id))
            node_id += 1

        # Função recursiva interna para ramificações fractais
        def add_fractal_branches(parent_id, px, py, angle, length, depth):
            nonlocal node_id
            if depth == 0: 
                return
            angles = [angle + np.pi/6, angle - np.pi/6]
            branch_len = length * 0.75
            for a in angles:
                nx = px + branch_len * np.cos(a)
                ny = py + branch_len * np.sin(a)
                curr_id = node_id
                nodes_data.append({'x': nx, 'y': ny})
                edges_raw.append((parent_id, curr_id))
                node_id += 1
                add_fractal_branches(curr_id, nx, ny, a, branch_len, depth - 1)

        # Aplicação das ramificações perpendiculares à espinha dorsal
        for s_id in spine_nodes[1:-1]:
            add_fractal_branches(s_id, nodes_data[s_id]['x'], nodes_data[s_id]['y'], np.pi/2, 3.0, self.levels)
            add_fractal_branches(s_id, nodes_data[s_id]['x'], nodes_data[s_id]['y'], -np.pi/2, 3.0, self.levels)

        # 2. Adição das Coletoras (Manifolds) de Saída
        df_temp = pd.DataFrame(nodes_data)
        y_max = df_temp['y'].max() + 1.0
        y_min = df_temp['y'].min() - 1.0

        all_indices = [e[0] for e in edges_raw] + [e[1] for e in edges_raw]
        counts = pd.Series(all_indices).value_counts()
        leaf_ids = counts[counts == 1].index.tolist()
        leaf_ids = [idx for idx in leaf_ids if idx not in [spine_nodes[0], spine_nodes[-1]]]

        for l_id in leaf_ids:
            target_y = y_max if nodes_data[l_id]['y'] > 0 else y_min
            new_id = node_id
            nodes_data.append({'x': nodes_data[l_id]['x'], 'y': target_y})
            edges_raw.append((l_id, new_id))
            node_id += 1

        # 3. Processamento Geométrico com Shapely para Interseções
        lines = [LineString([(nodes_data[e[0]]['x'], nodes_data[e[0]]['y']),
                             (nodes_data[e[1]]['x'], nodes_data[e[1]]['y'])]) for e in edges_raw]

        df_nodes_final = pd.DataFrame(nodes_data)
        for y_lim in [y_max, y_min]:
            pts = df_nodes_final[df_nodes_final['y'] == y_lim].sort_values('x')
            if len(pts) > 1:
                lines.append(LineString(pts[['x', 'y']].values))

        merged_graph = unary_union(lines)

        # 4. Mapeamento de IDs Únicos e Conversão para Arrays NumPy
        final_nodes_map = {}
        final_nodes_list = []
        final_edges_list = []

        def get_node_id(pt):
            coords = (round(pt[0], 6), round(pt[1], 6))
            if coords not in final_nodes_map:
                final_nodes_map[coords] = len(final_nodes_list)
                final_nodes_list.append([pt[0], pt[1]])
            return final_nodes_map[coords]

        segments = merged_graph.geoms if hasattr(merged_graph, 'geoms') else [merged_graph]
        for seg in segments:
            id_start = get_node_id(seg.coords[0])
            id_end = get_node_id(seg.coords[-1])
            final_edges_list.append([id_start, id_end])

        # Armazenamento e escalonamento para metros (SI)
        self.Xno = np.array(final_nodes_list) * self.coord_scale_to_m
        self.conec = np.array(final_edges_list)
        
        # Eliminação de auto-conexões espúrias (loops no mesmo nó)
        mask = self.conec[:, 0] != self.conec[:, 1]
        self.conec = self.conec[mask]
        
        self.nv = self.Xno.shape[0]
        self.nc = self.conec.shape[0]

    def _calcular_viscosidade_agua(self, T_celsius: float) -> float:
        """Calcula a viscosidade dinâmica padrão da água (Vogel-Tammann-Fulcher)."""
        A = 2.414e-5
        B = 247.8
        C = 140.0
        return A * 10 ** (B / ((T_celsius + 273.15) - C))

    def _calcular_viscosidade_empirica(self, T_celsius: float) -> float:
        """Calcula a viscosidade baseada na equação empírica do Tópico 6."""
        return 0.001791 / (1.0 + 0.03368 * T_celsius + 0.000221 * (T_celsius ** 2))

    def obter_propriedades_hidraulicas(self, t: float, mu_modo: str = "padrao") -> dict:
        """
        Determina as propriedades geométricas e a condutância dos canais.
        Suporta variação temporal da viscosidade.
        """
        # Seleção do modelo de viscosidade dinâmico ou fixo
        if self.mu_override is not None:
            mu = float(self.mu_override)
        elif mu_modo == "empirica":
            # Lei de aquecimento quadrática: T(t) = 20 + 0.9 * t^2
            T_t = 20.0 + 0.9 * (t ** 2)
            mu = self._calcular_viscosidade_empirica(T_t)
        else:
            mu = self._calcular_viscosidade_agua(self.temperature_celsius)

        # Cálculo vetorizado dos comprimentos das arestas (distância Euclidiana)
        x1 = self.Xno[self.conec[:, 0], :]
        x2 = self.Xno[self.conec[:, 1], :]
        L = np.linalg.norm(x2 - x1, axis=1)

        # Determinação vetorizada da área da seção transversal
        if self.area_per_edge is not None:
            area = np.array(self.area_per_edge, dtype=float)
        elif self.geometry_mode == "area":
            area = np.full(self.nc, self.area_constant, dtype=float)
        elif self.geometry_mode == "diameter":
            area = np.full(self.nc, np.pi * (self.diameter_constant ** 2) / 4.0, dtype=float)
        elif self.geometry_mode == "rectangular":
            area = np.full(self.nc, self.width * self.height, dtype=float)
        else:
            raise ValueError("Modo geométrico desconhecido.")

        # Diâmetro hidráulico equivalente e condutâncias (Hagen-Poiseuille)
        D_eq = np.sqrt(4.0 * area / np.pi)
        kappa = np.pi * (D_eq ** 4) / (128.0 * mu)
        C = kappa / L

        return {"mu": mu, "lengths": L, "areas": area, "conductances": C}

    def construir_matriz_incidencia(self) -> csr_matrix:
        """
        Monta a matriz de incidência topológica orientada D (nc x nv) 
        diretamente no formato esparso CSR.
        """
        # Instanciação direta em matriz esparsa via coordenadas (formato COO implícito)
        linhas = np.repeat(np.arange(self.nc), 2)
        colunas = self.conec.flatten()
        dados = np.tile([1.0, -1.0], self.nc) # 1 sai do nó i, -1 entra no nó j
        
        return csr_matrix((dados, (linhas, colunas)), shape=(self.nc, self.nv))

    def assembly_sistema_esparso(self, C: np.ndarray) -> csr_matrix:
        """
        Executa o Assembly vetorizado da matriz global de condutância A.
        Utiliza a identidade algébrica de grafos: A = D^T * K * D
        
        Decisão de projeto: Esta formulação elimina loops Python nativos e constrói
        a matriz A diretamente em formato esparso compacto, otimizando memória e CPU.
        """
        D = self.construir_matriz_incidencia()
        K = diags(C, format="csr") # Matriz diagonal esparsa de condutâncias
        
        # O produto de matrizes esparsas herda a estrutura correta de A
        A_esparsa = D.T @ K @ D
        return A_esparsa
    
    # =============================================================================
    # PARTE 5: SOLVERS E MOTORES NUMÉRICOS (MÉTODOS DOS TÓPICOS)
    # =============================================================================

    def _avaliar_sinal_vazao(self, spec: dict, t: float) -> float:
        """Avalia matematicamente o tipo de sinal de vazão imposto no instante t."""
        tipo = spec["type"].lower()
        if tipo == "constant":
            return float(spec["value"])
        
        mean = float(spec["mean"])
        amp = float(spec["amp"])
        freq = float(spec["freq"])
        fase = float(spec.get("phase", 0.0))
        
        if tipo == "sin":
            return mean + amp * np.sin(2.0 * np.pi * freq * t + fase)
        if tipo == "cos":
            return mean + amp * np.cos(2.0 * np.pi * freq * t + fase)
        
        raise ValueError(f"Tipo de sinal de vazão inválido: {tipo}")

    def construir_vetor_vazoes(self, t: float) -> np.ndarray:
        """Monta o vetor b global avaliando as condições de contorno de Neumann em t."""
        b = np.zeros(self.nv, dtype=float)
        for no, spec in self.flow_bc.items():
            if not (0 <= no < self.nv):
                raise ValueError(f"Nó {no} fora do limite da rede.")
            b[no] += self._avaliar_sinal_vazao(spec, t)
        return b

    def aplicar_condicoes_pressao(self, A: csr_matrix, b: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        """
        Aplica as condições de contorno de Dirichlet (pressão fixada).
        Zera a linha do nó restrito, define a diagonal como 1 e injeta o valor em b.
        """
        if len(self.pressure_bc) == 0:
            raise ValueError("Erro Numérico: É necessário ao menos uma pressão de contorno (Dirichlet).")
        
        A_mod = A.tolil()  # Conversão temporária para modificação eficiente de linhas
        b_mod = b.copy()
        
        for no, p_valor in self.pressure_bc.items():
            if not (0 <= no < self.nv):
                raise ValueError(f"Nó {no} inválido para restrição de pressão.")
            A_mod[no, :] = 0.0
            A_mod[no, no] = 1.0
            b_mod[no] = p_valor
            
        return A_mod.tocsr(), b_mod

    def resolver_instante(self, t: float, mu_modo: str = "padrao") -> dict:
        """Resolve a rede para um único instante t, retornando o estado completo."""
        prop = self.obter_propriedades_hidraulicas(t, mu_modo)
        A = self.assembly_sistema_esparso(prop["conductances"])
        b = self.construir_vetor_vazoes(t)
        
        A_mod, b_mod = self.aplicar_condicoes_pressao(A, b)
        
        # Resolução do sistema linear via solver esparso direto
        p = spsolve(A_mod, b_mod)
        
        # Pós-processamento matricial das vazões e quedas de pressão nas arestas
        D = self.construir_matriz_incidencia()
        q = diags(prop["conductances"], format="csr") @ (D @ p)
        dp = p[self.conec[:, 0]] - p[self.conec[:, 1]]
        
        return {"time": t, "p": p, "q": q, "dp_edge": dp, "mu": prop["mu"], "lengths": prop["lengths"], "areas": prop["areas"], "C": prop["conductances"], "b": b}

    def executar_topico_3_fluxo_por_pressao(self) -> None:
        """Tópico 3: Inverte a lógica impondo gradiente de pressão e calculando a vazão resultante."""
        print("\n" + "=" * 88)
        print("MÓDULO DE EXECUÇÃO: TÓPICO 3 (PROBLEMA DUAL - FLUXO INDUZIDO POR PRESSÃO)")
        print("=" * 88)
        
        # Configuração forçada do problema dual
        self.flow_bc = {}
        self.pressure_bc = {0: 100.0, 5: 0.0}
        
        res = self.resolver_instante(t=0.0, mu_modo="padrao")
        
        # Cálculo algébrico da vazão total que entra pelo Inlet (Nó 0)
        q_arestas = res["q"]
        vazao_inlet = 0.0
        for k, (i, j) in enumerate(self.conec):
            if i == 0:   vazao_inlet += q_arestas[k]
            elif j == 0: vazao_inlet -= q_arestas[k]
            
        print(f"Resultados obtidos para ΔP = 100 Pa:")
        print(f"  -> Pressão no Inlet (Nó 0): {res['p'][0]:.2f} Pa")
        print(f"  -> Pressão no Outlet (Nó 5): {res['p'][5]:.2f} Pa")
        print(f"  -> Vazão total calculada no Inlet: {vazao_inlet:.6e} m³/s ({vazao_inlet * 1e6:.4f} mL/s)")
        print("=" * 88 + "\n")

    def executar_topico_4_linearidade_transiente(self) -> None:
        """Tópico 4: Aceleração computacional baseada em resposta a uma carga unitária estática."""
        print("\n" + "=" * 88)
        print("MÓDULO DE EXECUÇÃO: TÓPICO 4 (LINEARIDADE E CARGA UNITÁRIA)")
        print("=" * 88)
        
        omega = 3.0
        times = np.linspace(0, 10, 1000)
        
        # Resolução da resposta à carga unitária (1 m³/s no nó 0)
        prop = self.obter_propriedades_hidraulicas(0.0, "padrao")
        A = self.assembly_sistema_esparso(prop["conductances"])
        b_unit = np.zeros(self.nv)
        b_unit[0] = 1.0
        
        A_mod, b_mod = self.aplicar_condicoes_pressao(A, b_unit)
        p_unit = spsolve(A_mod, b_mod)
        
        p_max_history = []
        print("Acelerando simulação transiente via escalonamento linear (Fatoração Única)...")
        for t in times:
            Q_t = (1.0 + 0.1 * np.sin(omega * t)) * 1e-6  # mL/s para m³/s
            p_t = Q_t * p_unit  # Superposição direta
            p_max_history.append(np.max(p_t))
            
        self._plotar_resultado_topico(times, p_max_history, "Tópico 4: Pressão Máxima vs Tempo", "darkblue")

    def executar_topico_5_superposicao_efeitos(self) -> None:
        """Tópico 5: Superposição linear multi-entrada usando duas soluções de base independentes."""
        print("\n" + "=" * 88)
        print("MÓDULO DE EXECUÇÃO: TÓPICO 5 (SUPERPOSIÇÃO DE MÚLTIPLAS ENTRADAS)")
        print("=" * 88)
        
        # Ajuste de segurança: impede quebra por índice inexistente em malhas pequenas
        no_secundario = 175
        if self.nv <= no_secundario:
            print(f"Risco de Bug de Projeto: Nó {no_secundario} não existe nesta malha (Apenas {self.nv} nós disponíveis).")
            no_secundario = self.nv - 1
            print(f"Correção Automática: Utilizando o nó {no_secundario} como segunda entrada.")
            
        omega = 4.0
        times = np.linspace(0, 10, 1000)
        
        prop = self.obter_propriedades_hidraulicas(0.0, "padrao")
        A = self.assembly_sistema_esparso(prop["conductances"])
        
        # Base 1: Carga unitária no Nó 0
        b_0 = np.zeros(self.nv); b_0[0] = 1.0
        A_mod, b_mod0 = self.aplicar_condicoes_pressao(A, b_0)
        p_unit0 = spsolve(A_mod, b_mod0)
        
        # Base 2: Carga unitária no Nó Secundário
        b_sec = np.zeros(self.nv); b_sec[no_secundario] = 1.0
        _, b_mod_sec = self.aplicar_condicoes_pressao(A, b_sec)
        p_unit_sec = spsolve(A_mod, b_mod_sec)
        
        p_max_history = []
        for t in times:
            Q0_t = (1.0 + 0.1 * np.sin(omega * t)) * 1e-6
            Qsec_t = (0.1 + 0.01 * np.cos(omega * t)) * 1e-6
            
            # Combinação linear das respostas estáticas
            p_t = (Q0_t * p_unit0) + (Qsec_t * p_unit_sec)
            p_max_history.append(np.max(p_t))
            
        self._plotar_resultado_topico(times, p_max_history, "Tópico 5: Superposição Linear Multi-Nó", "darkgreen")

    def executar_topico_6_viscosidade_variavel(self) -> None:
        """Tópico 6: Quebra da linearidade devido ao aquecimento progressivo. Exige remontagem completa da matriz."""
        print("\n" + "=" * 88)
        print("MÓDULO DE EXECUÇÃO: TÓPICO 6 (NÃO-LINEARIDADE TÉRMICA TRANSITÓRIA)")
        print("=" * 88)
        
        self.flow_bc = {0: {"type": "constant", "value": 1.0e-7}}
        self.pressure_bc = {5: 0.0}
        
        times = np.arange(self.t0, self.tf + self.dt, self.dt)
        p_max_history = []
        
        print("Aviso de Desempenho: Matriz A variante no tempo. Remontando e refatorando a cada passo...")
        for t in times:
            res = self.resolver_instante(t, mu_modo="empirica")
            p_max_history.append(np.max(res["p"]))
            
        self._plotar_resultado_topico(times, p_max_history, "Tópico 6: Impacto do Aquecimento Gradual", "darkred")

    def executar_topico_7_analise_complexidade(self) -> None:
        """Tópico 7: Benchmarking numérico comparando o tempo de montagem versus resolução esparsa."""
        print("\n" + "=" * 88)
        print("MÓDULO DE EXECUÇÃO: TÓPICO 7 (ANÁLISE DE DESEMPENHO COMPUTACIONAL)")
        print("=" * 88)
        print(f"{'Levels':<8} | {'Nº de Nós':<10} | {'Tempo Assembly (s)':<20} | {'Tempo Resolução (s)':<20}")
        print("-" * 67)
        
        config_base = {"geometry_mode": self.geometry_mode, "width": self.width, "height": self.height, "pressure_bc": {5: 0.0}, "flow_bc": {0: {"type": "constant", "value": 1e-7}}}
        
        for lvl in [1, 2, 3, 4, 5]:
            # Instanciação temporária para isolar o crescimento do grafo
            rede_teste = RedeHidraulica(config_base)
            rede_teste.levels = lvl
            rede_teste.gerar_topologia_grafo()
            
            prop = rede_teste.obter_propriedades_hidraulicas(0.0, "padrao")
            b_base = rede_teste.construir_vetor_vazoes(0.0)
            
            t_assembly_acum = 0.0
            t_solve_acum = 0.0
            
            # Média de 10 execuções para amortecer ruídos operacionais do sistema operacional
            for _ in range(10):
                t0 = time.perf_counter()
                A = rede_teste.assembly_sistema_esparso(prop["conductances"])
                A_mod, b_mod = rede_teste.aplicar_condicoes_pressao(A, b_base)
                t1 = time.perf_counter()
                
                t_assembly_acum += (t1 - t0)
                
                t2 = time.perf_counter()
                p = spsolve(A_mod, b_mod)
                t3 = time.perf_counter()
                
                t_solve_acum += (t3 - t2)
                
            print(f"{lvl:<8} | {rede_teste.nv:<10} | {t_assembly_acum/10:<20.6e} | {t_solve_acum/10:<20.6e}")
        print("=" * 88 + "\n")

    def _plotar_resultado_topico(self, x, y, titulo, cor):
        """Método utilitário para padronização de saídas gráficas isoladas."""
        plt.figure(figsize=(9, 4.5))
        plt.plot(x, y, color=cor, linewidth=2)
        plt.xlabel("Tempo [s]")
        plt.ylabel("Pressão Máxima na Rede [Pa]")
        plt.title(titulo)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
    
    # =============================================================================
    # PARTE 6: PÓS-PROCESSAMENTO, RELATÓRIOS E VISUALIZAÇÃO GRÁFICA
    # =============================================================================

    def calcular_potencia_dissipada(self, p: np.ndarray, C: np.ndarray) -> float:
        """
        Calcula a potência hidráulica total dissipada por atrito viscoso na rede.
        Formulação algébrica matricial: P = p^T * (D^T * K * D) * p = p^T * A * p
        """
        D = self.construir_matriz_incidencia()
        K = diags(C, format="csr")
        A_cond = D.T @ K @ D
        return float(p.T @ A_cond @ p)

    def calcular_residuos_massa(self, q: np.ndarray, b_imposto: np.ndarray) -> np.ndarray:
        """
        Verifica a consistência numérica do solver através da Lei de Conservação de Massa.
        O resíduo de cada nó deve tender a zero (ordem de convergência da máquina: ~1e-15).
        """
        residual = np.zeros(self.nv, dtype=float)
        
        # Acumulação topológica das vazões direcionadas pelas arestas
        for k, (i, j) in enumerate(self.conec):
            residual[i] += q[k]  # Fluxo saindo do nó i
            residual[j] -= q[k]  # Fluxo entrando no nó j
            
        return residual - b_imposto

    def exibir_resumo_entradas(self, prop: dict) -> None:
        """Exibe no terminal o sumário estruturado dos parâmetros de entrada (SI)."""
        print("=" * 88)
        print("CONFIGURAÇÕES DE ENTRADA DO GÊMEO DIGITAL (SUBSISTEMA HIDRÁULICO)")
        print("=" * 88)
        print(f"Níveis Fractais (levels): {self.levels}")
        print(f"Nós Gerados (nv): {self.nv} | Canais Gerados (nc): {self.nc}")
        print(f"Regime Temporal: {self.time_mode.upper()}")
        print(f"Intervalo de Simulação: [{self.t0}, {self.tf}] s | Passo (dt): {self.dt} s")
        print(f"Modo Geométrico dos Canais: {self.geometry_mode}")
        print(f"Viscosidade Dinâmica Operacional: {prop['mu']:.6e} Pa.s")
        print("\nPressões de Contorno (Dirichlet):")
        for no, val in self.pressure_bc.items():
            print(f"  -> Nó {no}: {val:.4e} Pa")
        print("\nVazões de Contorno (Neumann):")
        for no, spec in self.flow_bc.items():
            print(f"  -> Nó {no}: {spec}")
        print("=" * 88)

    def exibir_resumo_saidas(self, res: dict) -> None:
        """Apresenta os resultados numéricos processados no último passo de tempo."""
        p = res["p"]
        q = res["q"]
        
        potencia = self.calcular_potencia_dissipada(p, res["C"])
        residuos = self.calcular_residuos_massa(q, res["b"])
        
        print("\n" + "=" * 88)
        print(f"SAÍDAS DA SIMULAÇÃO (Instante Final: {res['time']:.4f} s)")
        print("=" * 88)
        print("Perfil de Pressões Nodais [Pa]:")
        for i, pi in enumerate(p):
            print(f"  -> Nó {i:3d}: {pi:.6e} Pa")
            
        print(f"\nPotência Hidráulica Dissipada: {potencia:.6e} W")
        
        print("\nResíduos da Conservação de Massa por Nó (Validação Numérica):")
        for i, ri in enumerate(residuos):
            print(f"  -> Nó {i:3d}: {ri:.6e} m³/s")
            
        print("\nTabela Detalhada de Comportamento dos Canais:")
        print(f"{'Canal':<5} | {'i -> j':<8} | {'Comprimento(m)':<14} | {'Área(m²)':<12} | {'Δp (Pa)':<12} | {'Vazão (m³/s)':<12}")
        print("-" * 75)
        for k, (i, j) in enumerate(self.conec):
            print(f"{k:5d} | {i:3d}->{j:<3d} | {res['lengths'][k]:.4e} | {res['areas'][k]:.4e} | {res['dp_edge'][k]:.4e} | {q[k]:.4e}")
        print("=" * 88)

    def plotar_mapa_pressao_rede(self, res: dict) -> None:
        """Gera a representação espacial 2D do grafo com mapa de cores para pressão."""
        p = res["p"]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')
        
        # Renderização dos nós coloridos de acordo com a magnitude da pressão
        cmap = plt.get_cmap("coolwarm")
        norm = plt.Normalize(vmin=float(p.min()), vmax=float(p.max()))
        cores_nos = [cmap(norm(pi)) for pi in p]
        
        ax.scatter(self.Xno[:, 0], self.Xno[:, 1], s=40, c=cores_nos, zorder=3, edgecolors="black")
        
        # Desenho das arestas e setas indicativas de direção do fluxo
        for k, (i, j) in enumerate(self.conec):
            x1, y1 = self.Xno[i, 0], self.Xno[i, 1]
            x2, y2 = self.Xno[j, 0], self.Xno[j, 1]
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=0.6, zorder=1)
            
            # Vetor direção para posicionamento da seta no ponto médio
            xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            p1, p2 = p[i], p[j]
            direcao = 1 if p1 > p2 else -1
            
            dx, dy = x2 - x1, y2 - y1
            L = np.hypot(dx, dy)
            if L > 0:
                ax.annotate("", xy=(xm + direcao * 0.1 * dx/L, ym + direcao * 0.1 * dy/L),
                            xytext=(xm - direcao * 0.1 * dx/L, ym - direcao * 0.1 * dy/L),
                            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8, mutation_scale=8), zorder=5)
                                
        ax.axis("off")
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label="Pressão Nodal, p [Pa]", fraction=0.03, pad=0.04)
        plt.title(f"Gêmeo Digital - Malha Hidráulica (t = {res['time']:.2f} s)")
        plt.tight_layout()
        plt.show()

    def plotar_series_temporais(self, historico: list, nos: list, arestas: list) -> None:
        """Plota a evolução dinâmica das pressões e vazões ao longo do tempo simulação."""
        tempos = np.array([r["time"] for r in historico])
        
        if nos:
            plt.figure(figsize=(9, 4))
            for no in nos:
                pressoes = np.array([r["p"][no] for r in historico])
                plt.plot(tempos, pressoes, label=f"Nó {no}")
            plt.xlabel("Tempo [s]"); plt.ylabel("Pressão [Pa]")
            plt.title("Histórico de Pressões Nodais Transientes")
            plt.grid(True, linestyle="--", alpha=0.5); plt.legend(); plt.tight_layout(); plt.show()
            
        if arestas:
            plt.figure(figsize=(9, 4))
            for art in arestas:
                vazoes = np.array([r["q"][art] for r in historico])
                plt.plot(tempos, vazoes, label=f"Canal {art}")
            plt.xlabel("Tempo [s]"); plt.ylabel("Vazão [m³/s]")
            plt.title("Histórico de Vazões nos Canais")
            plt.grid(True, linestyle="--", alpha=0.5); plt.legend(); plt.tight_layout(); plt.show()


# =============================================================================
# DICIONÁRIO DE CONFIGURAÇÃO DE TESTE PRINCIPAL
# =============================================================================

CONFIG_GLOBAL = {
    "levels": 3,
    "coord_scale_to_m": 1e-3,
    "time_mode": "transient",
    "t0": 0.0,
    "tf": 10.0,
    "dt": 0.2,
    "geometry_mode": "rectangular",
    "width": 500e-6,
    "height": 500e-6,
    "pressure_bc": {5: 0.0},
    "flow_bc": {
        0: {"type": "sin", "mean": 1.0e-7, "amp": 0.3e-7, "freq": 0.5, "phase": 0.0},
        3: {"type": "cos", "mean": -0.2e-7, "amp": 0.1e-7, "freq": 0.25, "phase": 0.0},
    },
    "temperature_celsius": 25.0,
    "mu_override": None,
    
    # Flags de ativação dos tópicos específicos de investigação do relatório
    "run_topic_3": False,
    "run_topic_4": False,
    "run_topic_5": False,
    "run_topic_6": False,
    "run_topic_7": False,
}

# =============================================================================
# FUNÇÃO ORQUESTRADORA PRINCIPAL (MAIN)
# =============================================================================

def main():
    """ Instancia o subsistema da rede hidráulica e gerencia a árvore de execução. """
    # Inicialização do objeto e construção da topologia
    rede = RedeHidraulica(CONFIG_GLOBAL)
    rede.gerar_topologia_grafo()
    
    # Desvios condicionais para a execução isolada dos tópicos de estudo do relatório
    if CONFIG_GLOBAL.get("run_topic_3"):
        rede.executar_topico_3_fluxo_por_pressao()
        return
    if CONFIG_GLOBAL.get("run_topic_4"):
        rede.executar_topico_4_linearidade_transiente()
        return
    if CONFIG_GLOBAL.get("run_topic_5"):
        rede.executar_topico_5_superposicao_efeitos()
        return
    if CONFIG_GLOBAL.get("run_topic_6"):
        rede.executar_topico_6_viscosidade_variavel()
        return
    if CONFIG_GLOBAL.get("run_topic_7"):
        rede.executar_topico_7_analise_complexidade()
        return

    # Execução Padrão: Loop Temporal Quase-Estático
    prop_iniciais = rede.obter_propriedades_hidraulicas(t=0.0, mu_modo="padrao")
    rede.exibir_resumo_entradas(prop_iniciais)
    
    times = np.arange(rede.t0, rede.tf + rede.dt, rede.dt)
    historico_resultados = []
    
    print("Iniciando laço temporal de resolução...")
    for t in times:
        res_t = rede.resolver_instante(t, mu_modo="padrao")
        historico_resultados.append(res_t)
        
    # Relatórios e saídas do último passo calculado
    ultimo_estado = historico_resultados[-1]
    rede.exibir_resumo_saidas(ultimo_estado)
    
    # Renderização Gráfica do Estado Espacial da Rede
    rede.plotar_mapa_pressao_rede(ultimo_estado)
    
    # Renderização Gráfica das Curvas Dinâmicas
    # Acompanha os nós [0, 2, 5] e canais [0, 1, 2] predefinidos para monitoramento
    rede.plotar_series_temporais(historico_resultados, nos=[0, 2, 5], arestas=[0, 1, 2])

if __name__ == "__main__":
    main()