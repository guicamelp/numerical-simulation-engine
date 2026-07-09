# Acoplamento Hidráulico-Mecânico (Interação Fluido-Estrutura)

Este repositório contém a implementação numérica de um modelo de Interação Fluido-Estrutura (FSI) fortemente acoplado. O código resolve a dinâmica transiente de uma membrana elástica circular excitada pelo escoamento incompressível em uma rede microfluídica fractal subjacente.

---

## 1. Intuição e Fundamentos

O sistema é composto por dois domínios físicos acoplados cinemática e dinamicamente:

* **Rede Hidráulica (1D):** Representada por um grafo fractal. O escoamento assume a aproximação de fluxo laminar de Poiseuille. O modelo elétrico análogo (Leis de Kirchhoff) é utilizado, onde as diferenças de pressão governam as vazões e a conservação de massa impõe divergente nulo nos nós (exceto na interface).
* **Estrutura (2D):** Uma placa/membrana elástica circular fina, governada pela equação da onda amortecida: $\rho e \frac{\partial^2 w}{\partial t^2} + \beta \frac{\partial w}{\partial t} - \sigma \nabla^2 w = p_{fluido}$. A discretização espacial é feita via Método das Diferenças Finitas (MDF).
* **Acoplamento Monolítico:** A vazão deslocada pela membrana atua como termo fonte no nó de descarga da rede hidráulica. As equações são resolvidas simultaneamente em uma única matriz global esparsa, mitigando instabilidades numéricas. A discretização temporal utiliza o método de Euler Implícito (Backward Euler).

### Sistema de Adimensionalização
Para evitar matrizes mal condicionadas devido à diferença de ordens de grandeza, o tempo e as variáveis de estado são adimensionalizados:

* **Tempo:** $\hat{t} = t / t_{ref}$, onde $t_{ref} = R \sqrt{\rho e / \sigma}$
* **Deflexão:** $\hat{w} = w / w_{ref}$, onde $w_{ref} = 0.01R$
* **Pressão:** $\hat{p} = p / p_{ref}$, onde $p_{ref} = \sigma w_{ref} / R^2$

---

## 2. Dependências e Instalação

As dependências são estritamente orientadas à computação científica e geometria.

``` bash
pip install numpy scipy pandas matplotlib shapely
```

---

## 3. Uso da Interface de Linha de Comando (CLI)

A execução é modularizada. Por padrão, todos os tópicos (1 a 5) são executados.

| Comando / Flag | Descrição | Exemplo de Uso |
| :--- | :--- | :--- |
| `--topics` | Especifica quais rotinas (1 a 5) executar. | `python main.py --topics 2 4` |
| `--quick` | Roda uma versão reduzida do Tópico 2 para testes rápidos. | `python main.py --quick` |
| `--full-sweep` | Roda a varredura paramétrica completa (96 casos) do Tópico 2. | `python main.py --topics 2 --full-sweep` |
| `--output` | Define o diretório raiz para os arquivos gerados (gráficos e CSVs). | `python main.py --output "resultados"` |

---

## 4. Decisões de Projeto Numérico (Trade-offs)

### A. Formulação Monolítica vs. Particionada
* **Decisão:** O código monta uma matriz global gigante acoplando hidráulica e mecânica (`Aglob`) e utiliza um solver direto esparso (fatoração LU via `scipy.sparse.linalg.splu`).
* **Trade-off:** Garante estabilidade incondicional (essencial para acoplamento forte devido à massa adicionada do fluido) e evita o custo de iterações em cada passo de tempo (necessárias na abordagem particionada). O revés é o alto consumo de memória RAM ($O(N^2)$ para a estrutura da matriz LU) limitando o refino da malha 2D.

### B. Geometria da Membrana (Fictitious Domain)
* **Decisão:** Utilizou-se uma malha cartesiana quadrada ($Nx \times Ny$) englobando a membrana circular, penalizando os nós externos adicionando um valor arbitrariamente grande ($10^8$) à diagonal da matriz de rigidez $K$.
* **Trade-off:** Facilita e vetoriza drasticamente a montagem de $K$ (estêncil regular de 5 pontos). A limitação é a representação em "degraus" do contorno circular contínuo (erro de discretização de fronteira de ordem $O(h)$), que exige malhas finas (ex: 101x101) para alta precisão.

### C. Condutância Hidráulica Equivalente
* **Decisão:** Os canais quadrados reais foram aproximados por canais circulares com o mesmo Diâmetro Hidráulico Equivalente ($D_{eq} = \sqrt{4A/\pi}$).
* **Trade-off:** Permite o uso direto da equação de Poiseuille fechada (evitando as séries infinitas da solução analítica do duto retangular), barateando a computação do grafo inicial com precisão aceitável para o balanço de massa sistêmico.

---

## 5. Descrição das Rotinas (Tópicos)

* **Tópico 1 (Matriz de Resistência R):** Visualiza a estrutura de esparsidade do complemento de Schur que projeta o fluido sobre a membrana. A inversão da matriz de condutância $A$ é evitada substituindo-a pela solução do sistema $AX = U$.
* **Tópico 2 (Evolução Paramétrica):** Marcha no tempo submetendo a membrana à pressão prescrita até o equilíbrio. Analisa a sensibilidade do sistema a variações de pressão ($p_{inlet}$), largura de canais ($H$), passos de tempo ($\Delta\hat{t}$) e refino de malha ($N$).
* **Tópico 3 (Queda de Pressão):** Problema de relaxamento elástico. A partir de um estado deformado de equilíbrio, remove-se a pressão externa ($p_{inlet}=0$) e observa-se a descarga reversa de escoamento.
* **Tópico 4 (Oscilação Livre):** Resolve o problema de autovalor generalizado ($K\phi = \lambda M\phi$). Inicializa o sistema estritamente na forma do 3º modo vibracional natural e avalia sua frequência por interpolação de cruzamentos por zero da resposta dinâmica amortecida transiente.
* **Tópico 5 (Forçamento Harmônico):** Submete a entrada de fluido a uma pressão oscilatória $P(t) = P_0 \cos(\omega_3 t)$, testando a ressonância do sistema sob forçamento externo na frequência natural mecânica do 3º modo.

---

## 6. Melhorias Futuras e Riscos Apontados

* **Gargalo de Memória:** Para malhas significativamente maiores que 101x101, a matriz monolítica estourará a memória. Solução sugerida: Implementar solvers iterativos com pré-condicionadores de bloco (ex: GMRES com pré-condicionador ILU).
* **Método de Penalização:** Penalizações muito altas induzem problemas de número de condição na matriz. Como contramedida no código atual, os nós externos também tiveram suas conexões Laplacianas removidas da montagem, minimizando ruído espúrio.
* **Não-Linearidade Geométrica:** O modelo mecânico assume pequenas deflexões. Para pressões altíssimas (onde deflexão > espessura), a teoria das membranas não lineares (tensões no plano de von Kármán) deve ser incorporada iterativamente (ex: Newton-Raphson).