# Gêmeo Digital: Subsistema de Rede Hidráulica

## 1. Visão Geral e Intuição Física
Este repositório contém o motor numérico para o subsistema hidráulico de um Gêmeo Digital de dispositivos microfluídicos. A intuição central do modelo é a abstração de um sistema de fluidos contínuo para um domínio discreto baseado em grafos:
* **Nós (Vértices):** Representam as conexões e uniões do sistema, armazenando a variável escalar de **pressão** ($p$).
* **Arestas (Canais):** Representam os dutos microfluídicos, conduzindo a variável vetorial de **vazão** ($Q$).

O objetivo principal é calcular o campo de pressões e vazões sob condições transientes e estacionárias, servindo como base para futuros acoplamentos com subsistemas térmicos e elásticos.

---

## 2. Fundamentos Matemáticos
A física do sistema repousa na formulação de equações algébricas lineares derivadas de dois princípios macroscópicos:

1.  **Lei Constitutiva (Hagen-Poiseuille):** A vazão ao longo de um canal laminar é proporcional ao gradiente de pressão:
    $$Q_{k} = C_{k}(p_{i}^{k} - p_{j}^{k})$$
    Onde $C_{k}$ é a condutância hidráulica do canal, função de sua geometria e da viscosidade dinâmica do fluido ($\mu$).
2.  **Conservação de Massa Nodal (1ª Lei de Kirchhoff para fluidos):** O somatório das vazões convergindo para qualquer nó sem acúmulo é estritamente zero:
    $$\sum Q_{k}^{(i)} = 0$$

Ao aplicar a conservação de massa em toda a topologia, o problema físico se reduz à resolução de um sistema linear global:
$$A p = b$$
Onde $A$ é a matriz global de condutância (construída via $D^{T} K D$), $p$ é o vetor de pressões desconhecidas e $b$ contém as restrições de fluxo (Neumann). A singularidade de $A$ é quebrada impondo-se pressões conhecidas nos contornos (Dirichlet).

---

## 3. Arquitetura e Decisões de Projeto
O código foi refatorado do paradigma procedural clássico para a **Orientação a Objetos (POO)**. O estado da simulação, matrizes topológicas e parâmetros físicos são restritos à classe `RedeHidraulica`.

### Trade-offs e Otimizações Numéricas
A tabela abaixo resume as principais decisões algorítmicas adotadas para garantir estabilidade e performance computacional:

| Componente | Abordagem Implementada | Trade-off / Justificativa |
| :--- | :--- | :--- |
| **Estrutura Matricial** | Matrizes Esparsas (`scipy.sparse`) | **A favor:** Evita estouro de RAM $O(n^2)$ e acelera soluções para grandes grafos. <br>**Contra:** Manipulação requer conhecimento explícito dos formatos (LIL para montagem, CSR para álgebra). |
| **Montagem (*Assembly*)** | Produto Matricial Direto ($A = D^{T} K D$) | **A favor:** Elimina gargalos de tempo ($O(n)$) causados por laços `for` em Python interpretado. Transfere o cálculo para *C/Fortran* no backend. |
| **Solver de Sistema** | Fatoração Esparsa Direta (`spsolve`) | **A favor:** Otimizado para não iterar sobre zeros, reduzindo a complexidade de tempo de $O(n^3)$ para soluções quase lineares. |
| **Simulação Transiente** | Superposição Linear (Matriz Constante) | **A favor:** Reduz o custo computacional avaliando a matriz unitária apenas uma vez, multiplicando pelo escalar do tempo. <br>**Limitação:** Inválido se a temperatura (e, consequentemente, a matriz $A$) mudar no tempo. |

---

## 4. Dependências e Instalação

O ambiente de execução requer as seguintes bibliotecas para cômputo científico, topologia e geração gráfica:

* **NumPy:** Vetorização e processamento algébrico de arrays contíguos.
* **SciPy:** Fornece o módulo `sparse` imperativo para montagem e resolução de malhas.
* **Pandas:** Tabulação auxiliar e análise rápida de conectividades.
* **Shapely:** Processamento geométrico booleano para interseção de coletoras (manifolds).
* **Matplotlib:** Visualização temporal 2D e renderização espacial do grafo.

### Comando de Instalação:
```bash
pip install numpy scipy pandas shapely matplotlib