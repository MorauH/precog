# AGENTS.md — precog

Hierarchical predictive-coding world model for vehicle control via ROS2.
_Online learning — no training/inference separation._

## Stack

| Component | Detail |
|-----------|--------|
| Language | Python 3.13+ |
| ML | PyTorch (CPU via `--extra cpu`, CUDA 13.2 via `--extra cuda`) |
| ROS | rclpy, rosidl_runtime_py (external dependency) |
| Data | numpy, mcap, mcap-ros2-support |
| Package mgr | uv + setuptools (`src` layout) |
| Nix | `flake.nix` dev shell with auto CPU/CUDA detection |

## Commands

### Install

```bash
uv sync --extra cpu     # CPU-only
uv sync --extra cuda    # CUDA 13.2 (mutually exclusive with cpu)
```

### Run

```bash
python src/precog/test_read_env.py   # Read-only telemetry smoke test
python -m precog.main                 # Legacy synchronous control loop
python -m precog.processes.cli        # Pipelined actor (decoupled learning)
precog                               # After install (entry point → pipelined)
```

### Lint & Format

No linting/formatter is configured in `pyproject.toml`. If added in future, run:

```bash
ruff check .
ruff format .
```

Ruff config belongs in `[tool.ruff]` in `pyproject.toml` or standalone `.ruff.toml`.

### Type Checking

No mypy config exists. If added, run:

```bash
mypy src/precog
```

### Testing

No test framework is configured yet. There are no test files, `tests/` directory, or
`pytest` config. When tests are added, use `pytest` and document commands here.

Single file execution pattern (used for smoke tests):

```bash
python src/precog/test_read_env.py
```

## Architecture

```
src/precog/
  envs/ros/
    codecs.py           Pluggable msg↔array codecs + cross-key transforms
    ros.py              ROSEnvironment Node (Gym-like reset/step API)
    env_config.yaml     Declarative topic mapping
    source_selector.py  Multi-source input selection
  model/
    model.py            HierarchicalPCWorldModel entry point
    pc_level.py         Single PC level with SSM
    ssm.py              Selective State Space Model (Mamba-like)
    config.py           ModelConfig, DEFAULT_CONFIG
    diagnostics.py      Online metrics collection
    hierarchical_clock.py  Multi-rate tick scheduling
    level_state.py      Per-level runtime state
    multi_rate_runner.py   Multi-frequency clock + tick orchestrator
  dashboard/
    server.py           FastAPI live monitoring server (lazy import)
  main.py               Full control loop (env + model + runner)
```

**Data flow**: `ROS topics → codec (msg→array) → snapshot → transforms (cross-key feature
engineering) → model`

## Code Style

### Imports

Use multi-block ordering with blank-line separators between blocks:

1. `from __future__ import annotations` (in model code; omit in main/env where not needed)
2. Standard library imports (alphabetical preferred but not enforced)
3. Third-party imports (numpy, torch, rclpy, yaml)
4. Relative imports from sibling modules (`from .config import`)
5. Absolute imports from the `precog` package (`from precog.envs import`)

```python
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import ModelConfig
from precog.envs import ROSEnvironment
```

### Formatting

No project-wide formatter config. Conventions observed in existing code:
- 4-space indentation
- Blank lines between top-level definitions (classes, functions)
- Blank lines between import blocks
- Maximum line length ~100 characters (not strict)
- Section separators: `# --------------------------------` comment blocks for major sections
- Trailing commas in multi-line dicts/lists/dataclasses

### Types

- `from __future__ import annotations` in model code (lazy evaluation, enables `|` union
  syntax without runtime cost)
- Full type annotations on all function signatures (parameters and return types)
- `Optional[...]` for nullable values; `... | None` where `from __future__` is active
- `TYPE_CHECKING` for circular import guards:
  ```python
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from .model import HierarchicalPCWorldModel
  ```
- Shape annotations as inline comments on return types: `# (batch, d_state)`, `# (B, T, d_input)`
- Class attributes annotated inline in `__init__`: `self._queue: queue.Queue[str] = queue.Queue()`

### Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `SelectiveSSM`, `MultiRateRunner`, `KeyboardReader` |
| Functions/methods | snake_case | `build_runner`, `init_hidden`, `should_update` |
| Variables | snake_case | `obs_dict`, `hidden_states`, `tick_count` |
| Private methods/attrs | `_leading_underscore` | `_read_loop`, `_make_obs_callback` |
| Private module-level | `_leading_underscore` | `_float`, `_build_qos` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_CONFIG`, `ENV_CONFIG_PATH`, `BLEND_HZ` |
| Enums | PascalCase class, UPPER_SNAKE values (via `auto()`) | `OperationMode.DRIVE` |
| Dataclass fields | snake_case | `d_representation`, `level_frequencies` |
| Module filenames | snake_case | `multi_rate_runner.py`, `pc_level.py` |

### Error Handling

- `ValueError` for invalid arguments/config
- `assert` for invariant checks in `__post_init__` (e.g. `assert len(self.level_configs) >= 2`)
- `TimeoutError` for ROS readiness timeouts
- `try: ... except Exception: pass` for graceful fallback (e.g. torch.compile probes)
- `try: ... finally: ...` for resource cleanup (dashboard, env, keyboard)
- `try: queue.Empty` for non-blocking queue reads
- No bare `except:` — always catch a specific exception type

### Docstrings

Google-style with Args/Returns is preferred (used in most model code):

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Input tensor, shape (B, d_input).
    Returns:
        Output tensor, shape (B, d_output).
    """
```

- Shape annotations in docstrings: `shape (B, d_state)`
- One-liner docstrings for trivial accessors: `"""Zero-initialised SSM hidden state."""`
- ASCII art diagrams in module-level docstrings for architecture documentation

### Patterns

- **Dataclasses** for all config objects (no sentinel defaults, every field has a value)
- **Registry decorators** for extensibility: `@obs_codec("name")`, `@transform("name")`,
  `@action_codec("name")`. Register functions into module-level dicts. No changes to
  `ros.py` needed for new codecs or transforms.
- **nn.Module** subclass for all ML components
- **Device management**: explicit `device` parameter propagation, no hidden globals
- **Thread safety**: `threading.Lock()` with `with self._lock:` for shared state
- **Lazy imports** for optional deps (FastAPI, Dash components in dashboard code)

## Key Semantic Conventions

- **Codecs** are stateless 1:1 `msg → np.ndarray`. **Transforms** run post-snapshot with
  access to all keys, for cross-topic preprocessing.
- Observations may be `None` in the snapshot (stale/missing) — downstream encoders handle
  masking, not the bridge layer.
- Model expects observations as `dict[str, Optional[torch.Tensor]]` with batch dim 0.
- The system is online — `reset`, `step` (action), `tick` mental model, not
  train/eval/inference split.
