# Gêmeo Digital Multiphysics — Parte 6

## Escopo executado

Este pacote resolve integralmente os itens do capítulo enviado: Monte Carlo estacionário e dinâmico (Seção 6.3.2), interpolação/regressão e ruído (Seção 6.4.3), diferenciação numérica e sensibilidades (Seção 6.4.5), Newton–Raphson para a restrição energética e a animação integrada do desafio final. O fluxo é estritamente sequencial: **placa térmica → viscosidade por canal → rede hidráulica → membrana**, como definido no PDF.

## Arquitetura e álgebra linear

A placa usa a discretização conservativa já fornecida em `ThermalPlateSolver`, formando o sistema esparso `A_T T=b_T`. A geometria da rede modifica `k(x,y)` por proximidade dos canais através de `NetworkInfluenceModel`. Como a matriz térmica não depende de `T_C`, ela é fatorada uma vez e reutilizada nas varreduras de sensibilidade.

A viscosidade de cada aresta é calculada por quadratura de Newton–Cotes (regra composta do trapézio):

`μ̄_k = Σ_i w_i μ(T(x_k(s_i)))`.

Assim, a condutância de Poiseuille equivalente satisfaz `C_k ∝ H^4/μ̄_k`. A rede hidráulica é a Laplaciana ponderada do grafo `A_h=B diag(C) Bᵀ` e a etapa fluido-estrutura usa o sistema monolítico `G Y^(n+1)=F^(n+1)`, com `Y=[w,v,p]ᵀ`. A fatoração `SolveG=factorized(G)` é feita antes do laço temporal. Isso cumpre a otimização pedida no enunciado e permite resolver também o sistema de sensibilidades com a mesma fatoração.

## Resultado nominal

- Temperatura máxima da placa: **40.7730 °C**.
- Temperatura média da placa: **31.5455 °C**.
- Viscosidade média integrada nas arestas: **0.75731 mPa·s**.
- Energia adimensional no horizonte `[0,4]`: **5.38801989**.
- Vazão de entrada final: **133.106840 µL/s**.
- Volume acumulado final: **351.545027 nL**.

## Exercício 6.3.2 — Monte Carlo

Para cada cenário, `RandomFail(C_original,p_o,f_obs)` gera uma máscara Bernoulli independente usando `rng.random()`. As arestas obstruídas recebem `C_k/f_obs`; a probabilidade crítica é a média do indicador de falha. A convergência segue a ordem assintótica `N^(-1/2)` e o gráfico inclui uma banda de confiança de 95%.

- Caso estacionário de convergência: `p_o=0.40`, `f_obs=10`, estimativa final **0.656000**, estabilização operacional em **N≈732**.
- Caso dinâmico, `δt=0.05`: `Prob(E<7)=1.000000 ± 0.000000`.
- Caso dinâmico, `δt=0.10`: `Prob(E<7)=1.000000 ± 0.000000`.

A diferença entre passos temporais combina dois efeitos: a quadratura temporal altera os valores de energia por amostra e, portanto, alguns cenários podem cruzar o limiar de `E=7`; além disso, a incerteza estatística permanece da ordem `sqrt(p(1-p)/N)`. Por isso, refinar `δt` reduz viés de discretização, mas não substitui aumentar `N`.

## Exercício 6.4.3 — Aproximação e regressão

As splines lineares e cúbicas interpolam os pontos discretos. A cúbica preserva suavidade de primeira derivada nas interfaces, enquanto a linear é mais robusta perto de alterações abruptas. Para a regressão global, foi montada a matriz de Vandermonde escalada e resolvido o sistema normal `Aᵀ A c=Aᵀ y`; o erro é medido por `||r||_L2` via a regra do trapézio. O menor erro observado ocorreu no grau **15** no conjunto avaliado; os graus maiores devem ser interpretados junto à curvatura visual, pois a redução residual pode coexistir com oscilação de sobreajuste.

No experimento ruidoso, as splines passam exatamente pelos dados perturbados e, por isso, não filtram o ruído. A regressão de baixo/médio grau atua como filtro global e evidencia o compromisso entre fidelidade local e robustez.

## Exercício 6.4.5 — Sensibilidades

Foram calculadas diferenças progressivas e centradas em `T_C∈[0,250] °C` e `H∈[500,1500] µm`. Para `H`, o método direto usa

`dA/dH = (4/H)A` e resolve, em cada passo, `G S = [s_w/dt, M s_v/dt, -(dÃ/dH)p]ᵀ`.

As expressões solicitadas para as saídas são:

- `dq_in/dH = e_inᵀ[(dA/dH)p + A s_p]`;
- `dV(tf)/dH = ∫_0^tf h_hat² 1ᵀ s_v(t) · (v_ref R² t_ref) dt`.

Os gráficos `10–12` mostram as funções, forward, centered e a resposta direta. A diferença centrada apresenta erro de truncamento de ordem `O(ε²)`; o método direto não precisa de duas simulações completas por perturbação e é menos vulnerável à subtração de números próximos.

## Newton–Raphson para E(H)=7.5

Não existe mudança de sinal de E(H)-7.5 na varredura [100,2200] µm; o alvo é inviável sob o conjunto nominal de parâmetros.

O gráfico `13_newton_raphson_energia.png` registra a varredura de viabilidade e, quando há mudança de sinal, as iterações de Newton usando a derivada contínua. Caso não exista raiz para o conjunto nominal, isso é reportado explicitamente em vez de forçar uma solução numérica não física.

## Arquivos principais

- `gemeo_digital_integrado.py`: implementação completa e reprodutível.
- `resultados_gemeo_digital/*.png`: gráficos de todos os exercícios.
- `resultados_gemeo_digital/*.csv`: séries e tabelas numéricas.
- `resultados_gemeo_digital/14_animacao_gemeo_digital_3D.gif`: desafio de integração visual.
