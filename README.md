# local-ai-assistant

Asistente local en Python con separación básica entre:

- chat (`main.py`)
- modelo local (`llm/`)
- herramientas (`tools/`)

## Instalar dependencias

```bash
cd local-ai-assistant
pip install -r requirements.txt
```

## Configurar el modelo GGUF

Puedes usar cualquiera de estas opciones:

1. Definir `QWEN_MODEL_PATH` con la ruta absoluta al archivo `.gguf`
2. Guardar el modelo dentro de `local-ai-assistant/models/`
3. Guardar el modelo dentro de `~/local-ai-workspace/models/`

Ejemplo:

```bash
export QWEN_MODEL_PATH="/ruta/al/modelo/qwen.gguf"
```

Ruta detectada automáticamente en este entorno:

```bash
/Users/oswaldohernandez/local-ai-workspace/models/qwen2.5-3b/qwen2.5-3b-instruct-q5_k_m.gguf
```

## Ejecutar

Modo CLI:

```bash
python3 main.py chat "Explain what Snowflake is"
```

Modo interactivo:

```bash
python3 main.py
```
