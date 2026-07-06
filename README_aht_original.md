# Acoplamento Hidráulico-Térmico Integrado

## Intuição e Definição do Problema

Este projeto implementa a modelagem numérica de um sistema físico onde um domínio térmico (condução bidimensional) interage com um domínio hidráulico (escoamento unidimensional em rede de microcanais). A formulação baseia-se nas seguintes leis físicas de conservação:

1.  **Domínio Térmico:** Equação de Poisson para condução de calor em estado estacionário:
    -div(k grad T) = S
    Onde k é a condutividade térmica e S é a fonte volumétrica de calor. O problema é discretizado via Método de Volumes Finitos (FVM) em uma malha estruturada.

2.  **Domínio Hidráulico:** Conservação de massa em uma topologia de grafos (Leis de Kirchhoff aplicadas a fluidos) combinada com a aproximação de Hagen-Poiseuille para escoamento laminar em dutos:
    q_k = C_k * delta_p_k
    Onde a condutância C_k depende diretamente da viscosidade dinâmica do fluido mu(T).

**O Acoplamento:**
A interação é estabelecida pelo fato de que a viscosidade da água cai de forma não-linear com o aumento da temperatura. Canais mais quentes apresentam menor resistência ao escoamento. Em contrapartida, a presença da rede hidráulica altera a condutividade térmica efetiva da placa e atua como um sorvedouro localizado de energia.

---

## Fundamentos e Arquitetura

O código é modularizado para separar a resolução algébrica da física do problema:

| Componente | Função | Técnica Numérica Utilizada |
| :--- | :--- | :--- |
| `ThermalPlateSolver` | Resolve o campo contínuo T(x,y). | FVM com matriz Laplaciana esparsa e formato CSR. Solução via `scipy.sparse.linalg.spsolve`. |
| `NetworkInfluenceModel` | Mapeia a interação espacial entre malha e grafo. | Particionamento espacial via Árvore k-d (`scipy.spatial.cKDTree`) para consultas em O(log N). |
| `TemperatureInterpolator` | Transfere dados térmicos para o domínio fluido. | Splines bicúbicas e interpolação bilinear via `scipy.interpolate`. |
| Quadraturas em Arestas | Integra propriedades ao longo do canal. | Regras compostas do Ponto Médio e do Trapézio. |
| `HydroThermalCoupledModel` | Orquestra o fluxo de dados entre os solvers. | Padrão de projeto *Facade*. |

---

## Trade-offs e Limitações

* **Resolução da Malha vs. Custo Computacional:** A utilização de malhas densas (ex: 241x121) diminui o erro de discretização espacial (O(h^2)), mas aumenta o tamanho do sistema linear. O uso de matrizes esparsas mitiga o problema de alocação de memória (reduzindo de O(N^2) para O(N)), mas o custo de solução ainda escala de forma superlinear.
* **Aproximação de Hagen-Poiseuille:** A equação da condutância assume dutos circulares. O uso de "diâmetro equivalente" para microcanais de seção quadrada (500 µm x 500 µm) introduz um erro sistemático moderado devido aos efeitos de canto (tensões de cisalhamento não-uniformes).
* **Acoplamento One-Way vs. Two-Way Iterativo:** O modelo avalia a influência geométrica primeiro, resolve a temperatura, e impõe o resultado na hidráulica de forma sequencial. Não há um loop de convergência iterativo (ex: Picard ou Newton-Raphson) entre as duas físicas, o que significa que reações de feedback de segunda ordem são desprezadas.
* **Integração Direta da Viscosidade:** Pela Desigualdade de Jensen, como mu(T) é não-linear, avaliar a viscosidade na temperatura média (mu(<T>)) introduz erro em gradientes altos. A integração direta ao longo da aresta (<mu(T)>) fornece a resistência física exata do canal.

---

## Instalação e Dependências

Certifique-se de estar utilizando um ambiente Python (recomenda-se >= 3.8). Instale as dependências rigorosamente associadas às operações vetoriais e de manipulação de dados:

``` bash
pip install numpy scipy matplotlib pandas tabulate shapely
```

*Nota: O script suporta o carregamento condicional dos arquivos complementares do docente (`gera_grafo.py` e `plota_rede.py`). Se não forem encontrados no mesmo diretório, o modelo executará um gerador topológico de fallback.*

---

## Uso e Configuração

Para executar as simulações descritas nos exercícios, basta rodar o arquivo principal. O fluxo lógico interno é controlado pelas flags booleanas localizadas no dicionário `CONFIG_INTEGRADO`.

[código] python acoplamento_hidraulico_termico_integrado.py

As execuções principais habilitadas por padrão no `CONFIG_INTEGRADO` cobrem:
1. Dedução de regras de quadratura compostas.
2. Interpolação térmica multirresolução.
3. Acoplamento termo-hidráulico (convergência de quadraturas e cálculo de <mu>).
4. Sensibilidade da condutividade térmica à proximidade dos canais (d_max).
5. Modelação da fonte/sumidouro gaussiano a partir das intensidades da rede.

---

## Saídas Geradas

O modelo opera de forma idempotente, garantindo a criação segura do diretório `./resultados_acoplamento` e alocando todos os artefatos de saída ali.

| Artefato Gerado | Descrição | Formato |
| :--- | :--- | :--- |
| **Mapas de Contorno** | Campo térmico T(x,y) com projeção do grafo hidráulico. | `.png` |
| **Grafos Coloridos** | Avaliação térmica nos nós e arestas da rede. | `.png` |
| **Perfis 1D** | Cortes verticais e horizontais em regiões críticas (ex: y = Ly/2). | `.png` |
| **Tabelas Analíticas** | Erro numérico de quadratura, pressão máxima/mínima e potência dissipada. | `.csv` |
| **Relatório Integrado** | Síntese textual consolidada das métricas e respostas conceituais. | `.md` |