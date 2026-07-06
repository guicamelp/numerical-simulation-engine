# Como executar

```bash
pip install numpy scipy pandas matplotlib shapely pillow
python gemeo_digital_integrado.py --full
```

Use `--quick` para uma execução reduzida de validação. A execução `--full` usa 2000 realizações para cada caso dinâmico exigido pelo enunciado. Os resultados são gravados em `resultados_gemeo_digital/`.

O arquivo preserva e importa as versões anteriores de `acoplamento_hidraulico_termico.py` e `acoplamento_hidraulico_mecanico.py`; por isso, mantenha os três scripts na mesma pasta.
