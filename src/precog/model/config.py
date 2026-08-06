from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PCLevelConfig:
    d_representation: int
    d_below: Optional[int] = None
    d_above: Optional[int] = None
    predictor_hidden: List[int] = field(default_factory=lambda: [64])
    encoder_hidden: List[int] = field(default_factory=lambda: [64])
    sigreg_tau: float = 0.999
    sigreg_var_threshold: float = 0.1
    objective_enabled: bool = False
    objective_observable_key: str = ""
    objective_target_value: float = 0.0
    objective_ae_weight: float = 1.0
    objective_task_weight: float = 0.1
    translator_hidden: List[int] = field(default_factory=lambda: [16])


@dataclass
class ControlHeadConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    output_dim: int = 1
    output_scales: List[float] = field(default_factory=list)


@dataclass
class ModelConfig:
    observation_keys: List[str]
    control_dim: int
    d_input: int
    level_configs: List[PCLevelConfig]
    control_head: ControlHeadConfig = field(default_factory=ControlHeadConfig)
    control_level_idx: int = 1
    imitation_loss_weight: float = 1.0
    observation_shapes: Dict[str, tuple] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert len(self.level_configs) >= 2, "Need at least 2 PC levels"
        assert 0 <= self.control_level_idx < len(self.level_configs)


# ---------------------------------------------------------------------------
# Config resolution — bridges env_config.yaml and ModelConfig
# ---------------------------------------------------------------------------


def _env_shapes_from_yaml(env_cfg: dict) -> Dict[str, tuple]:
    shapes: Dict[str, tuple] = {}
    for obs in env_cfg.get("observations", []):
        shapes[obs["key"]] = tuple(obs["shape"])
    for key, out_cfg in env_cfg.get("transform_output_keys", {}).items():
        shapes[key] = tuple(out_cfg["shape"])
    return shapes


def resolve_config_dims(
    config: ModelConfig,
    env_obs_shapes: Dict[str, tuple],
) -> ModelConfig:
    d_input = 0
    for key in config.observation_keys:
        if key not in env_obs_shapes:
            raise ValueError(
                f"Observation key '{key}' not found in env observation shapes. "
                f"Available keys: {sorted(env_obs_shapes.keys())}"
            )
        shape = env_obs_shapes[key]
        assert shape[-1] > 0, f"shape[-1] must be positive, got {shape}"
        d_input += shape[-1]

    d_input += config.control_dim

    num_levels = len(config.level_configs)
    level_configs = list(config.level_configs)
    for i, lvl_cfg in enumerate(level_configs):
        d_below = d_input if i == 0 else level_configs[i - 1].d_representation
        d_above = level_configs[i + 1].d_representation if i + 1 < num_levels else None
        level_configs[i] = PCLevelConfig(
            d_representation=lvl_cfg.d_representation,
            d_below=d_below,
            d_above=d_above,
            predictor_hidden=lvl_cfg.predictor_hidden,
            encoder_hidden=lvl_cfg.encoder_hidden,
            sigreg_tau=lvl_cfg.sigreg_tau,
            sigreg_var_threshold=lvl_cfg.sigreg_var_threshold,
            objective_enabled=lvl_cfg.objective_enabled,
            objective_observable_key=lvl_cfg.objective_observable_key,
            objective_target_value=lvl_cfg.objective_target_value,
            objective_ae_weight=lvl_cfg.objective_ae_weight,
            objective_task_weight=lvl_cfg.objective_task_weight,
            translator_hidden=lvl_cfg.translator_hidden,
        )

    return ModelConfig(
        observation_keys=config.observation_keys,
        control_dim=config.control_dim,
        d_input=d_input,
        level_configs=level_configs,
        control_head=config.control_head,
        control_level_idx=config.control_level_idx,
        imitation_loss_weight=config.imitation_loss_weight,
        observation_shapes={k: env_obs_shapes[k] for k in config.observation_keys},
    )


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = ModelConfig(
    observation_keys=[
        "current_steering",
        "best_path_relative_sampling",
        "forward_speed",
        # "car_pose",
    ],
    control_dim=2,
    d_input=0,  # resolved at runtime from env shapes
    level_configs=[
        PCLevelConfig(
            d_representation=32,
            objective_enabled=True,
            objective_observable_key="forward_speed",
            objective_target_value=0.25,
            objective_ae_weight=1.0,
            objective_task_weight=0.1,
        ),
        PCLevelConfig(
            d_representation=32,
            objective_enabled=True,
            objective_observable_key="lateral_deviation",
            objective_target_value=0.0,
            objective_ae_weight=1.0,
            objective_task_weight=0.1,
        ),
        PCLevelConfig(
            d_representation=64,
            objective_enabled=False,
        ),
    ],
    control_head=ControlHeadConfig(
        hidden_dims=[16, 8], output_dim=2, output_scales=[0.5, 15.0]
    ),
    control_level_idx=0,
)
