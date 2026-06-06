# local-ai-assistant

A local Python assistant with basic separation between:

- chat (`main.py`)
- local model (`llm/`)
- tools (`tools/`)

## Install dependencies

```bash
cd local-ai-assistant
pip install -r requirements.txt
```

## Configure the GGUF model

You can use any of these options:

1. Set `QWEN_MODEL_PATH` to the absolute path of the `.gguf` file
2. Store the model under `local-ai-assistant/models/`
3. Store the model under `~/local-ai-workspace/models/`

Example:

```bash
export QWEN_MODEL_PATH="/path/to/qwen.gguf"
```

Automatically detected path in this environment:

```bash
/Users/oswaldohernandez/local-ai-workspace/models/qwen2.5-3b/qwen2.5-3b-instruct-q5_k_m.gguf
```

## Run

CLI mode:

```bash
python3 main.py chat "Explain what Snowflake is"
```

Interactive mode:

```bash
python3 main.py
```
