# local-ai-assistant

Estructura inicial de un asistente local en Python con separación básica entre:

- chat (`main.py`)
- modelo local (`llm/`)
- herramientas (`tools/`)

## Ejecutar

```bash
cd local-ai-assistant
python main.py
```

## Estructura

```text
local-ai-assistant/
  main.py
  requirements.txt
  README.md
  tools/
    __init__.py
    web_search.py
  llm/
    __init__.py
    qwen_runner.py
```

## Notas

- `llm/qwen_runner.py` contiene un runner simulado para el modelo local.
- `tools/web_search.py` define una herramienta base de búsqueda web.
