from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModalityConfig:
    """
    Configuration for a single sensor modality encoder (Level 0).

    Attributes:
        name:       Observation key in the env snapshot dict, e.g.
                    "imu", "car_pose", "best_path_relative_sampling".
        input_dim:  Raw feature dimension (last dimension of the tensor
                    shape produced by the codec + transform pipeline).
                    When ``None``, this is resolved at construction time
                    from the env config's shape declarations so the two
                    configs cannot drift apart.
        output_dim: Projected latent dimension fed into Level 1.
        hidden_dims: Optional hidden layers in the encoder FNN.
                     Empty list = single linear projection.
    """

    name: str
    input_dim: Optional[int] = None
    output_dim: int = 0
    hidden_dims: List[int] = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        return self.input_dim is not None


@dataclass
class SSMConfig:
    """
    Configuration for one SelectiveSSM inside a PC level.

    Attributes:
        d_state:        Internal SSM state dimension (memory capacity).
        dt_min / dt_max:  Timescale Δ range (min for fast dynamics,
                          max for slow dynamics).

    d_input and d_output are inferred from the PCLevelConfig context.
    """

    d_state: int
    dt_min: float = 0.001
    dt_max: float = 0.1


@dataclass
class PCLevelConfig:
    """
    Configuration for one level of the PC hierarchy.

    Attributes:
        d_representation:       Output dimension of this level's SSM.
                                Also the dimension of top-down predictions
                                received from the level above.
        ssm:                    SSM configuration for this level.
        prediction_head_hidden: Hidden dims of the FNN that predicts
                                the level below's representation.
        forward_head_hidden:    Hidden dims of the FNN that predicts this
                                level's own representation one step ahead
                                (used by the control head for anticipatory action).
    """

    d_representation: int
    ssm: SSMConfig
    prediction_head_hidden: List[int] = field(default_factory=lambda: [64])
    forward_head_hidden: List[int] = field(default_factory=lambda: [64])


@dataclass
class ControlHeadConfig:
    """
    Configuration for the control output head.

    Reads from [z_mid_t || ẑ_mid_{t+1}] and produces actions.

    Attributes:
        hidden_dims: Hidden layer dimensions of the control FNN.
        output_dim:  Number of control outputs (e.g. 2: steering + throttle).
    """

    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    output_dim: int = 2


@dataclass
class ModelConfig:
    """
    Full model configuration.

    Attributes:
        modalities:           Level 0 modality encoder configs.
                              The concatenated output dims must equal
                              level_configs[0]'s implied d_below.
        level_configs:        PC hierarchy levels, ordered low (fast) to
                              high (slow).  Exactly 3 levels expected.
        control_head:         Control output head config.
        control_level_idx:    Which PC level index feeds the control head
                              (default 1 = mid-level, 0-indexed).
        imitation_loss_weight: Relative weight of imitation vs. PC loss.
        pc_loss_weight:        Relative weight of PC prediction error loss.
    """

    modalities: List[ModalityConfig]
    level_configs: List[PCLevelConfig]
    control_head: ControlHeadConfig = field(default_factory=ControlHeadConfig)
    control_level_idx: int = 1
    imitation_loss_weight: float = 1.0
    pc_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        assert len(self.level_configs) >= 2, "Need at least 2 PC levels"
        assert 0 <= self.control_level_idx < len(self.level_configs)

    @property
    def is_resolved(self) -> bool:
        return all(m.is_resolved for m in self.modalities)

    @property
    def d_level0(self) -> int:
        if not self.is_resolved:
            raise RuntimeError(
                "d_level0 requires all modality input_dim values to be "
                "resolved.  Call resolve_config_dims() first."
            )
        return sum(m.output_dim for m in self.modalities)


# ---------------------------------------------------------------------------
# Config resolution — bridges env_config.yaml and ModelConfig
# ---------------------------------------------------------------------------


def _env_shapes_from_yaml(env_cfg: dict) -> Dict[str, tuple]:
    """Extract {key: shape_tuple} from a parsed env_config.yaml dict.

    Merges observation shapes and transform-output shapes into a single map.
    """
    shapes: Dict[str, tuple] = {}

    for obs in env_cfg.get("observations", []):
        s = tuple(obs["shape"])
        shapes[obs["key"]] = s

    for key, out_cfg in env_cfg.get("transform_output_keys", {}).items():
        shapes[key] = tuple(out_cfg["shape"])

    return shapes


def _last_dim_from_shape(shape: tuple) -> int:
    """Return the last element of *shape*, which is the input_dim for a
    modality encoder consuming this key."""
    assert shape[-1] > 0, f"shape[-1] must be a positive integer, got {shape[-1]}"
    return shape[-1]


def resolve_config_dims(
    config: ModelConfig,
    env_obs_shapes: Dict[str, tuple],
) -> ModelConfig:
    """Return a copy of *config* with every modality's ``input_dim`` resolved.

    ``env_obs_shapes`` is ``{observation_key: shape_tuple}`` from the env
    config (observations + transform outputs).

    - ``input_dim=None`` → resolved from ``env_obs_shapes[mod.name][-1]``.
    - ``input_dim`` explicitly set AND key is in env shapes → validated.
    - ``input_dim`` explicitly set AND key is NOT in env shapes → used as-is
      (internal modality, e.g. "control" which is injected by the runner).

    Raises :class:`ValueError` for missing auto-resolve keys or mismatches.
    """
    resolved_modalities: List[ModalityConfig] = []
    for mod in config.modalities:
        if mod.name not in env_obs_shapes:
            if mod.input_dim is not None:
                resolved_modalities.append(mod)
                continue
            raise ValueError(
                f"Modality '{mod.name}' not found in env observation shapes "
                f"and no explicit input_dim set. "
                f"Available keys: {sorted(env_obs_shapes.keys())}"
            )

        shape = env_obs_shapes[mod.name]
        dim_from_env = _last_dim_from_shape(shape)

        if mod.input_dim is None:
            resolved = dim_from_env
        elif mod.input_dim != dim_from_env:
            raise ValueError(
                f"Modality '{mod.name}': configured input_dim={mod.input_dim} "
                f"does not match env shape {shape} (last dim = {dim_from_env})"
            )
        else:
            resolved = mod.input_dim

        resolved_modalities.append(
            ModalityConfig(
                name=mod.name,
                input_dim=resolved,
                output_dim=mod.output_dim,
                hidden_dims=mod.hidden_dims,
            )
        )

    return ModelConfig(
        modalities=resolved_modalities,
        level_configs=config.level_configs,
        control_head=config.control_head,
        control_level_idx=config.control_level_idx,
        imitation_loss_weight=config.imitation_loss_weight,
        pc_loss_weight=config.pc_loss_weight,
    )


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = ModelConfig(
    modalities=[
        ModalityConfig(name="current_steering", output_dim=1, hidden_dims=[]),
        ModalityConfig(
            name="best_path_relative_sampling", output_dim=4, hidden_dims=[32]
        ),
        ModalityConfig(name="car_pose", output_dim=8, hidden_dims=[]),
        ModalityConfig(
            name="control",
            input_dim=2,
            output_dim=4,
            hidden_dims=[],
        ),
    ],
    level_configs=[
        PCLevelConfig(
            d_representation=32,
            ssm=SSMConfig(d_state=16, dt_min=0.001, dt_max=0.01),
            prediction_head_hidden=[16],
            forward_head_hidden=[16],
        ),
        PCLevelConfig(
            d_representation=32,
            ssm=SSMConfig(d_state=16, dt_min=0.01, dt_max=0.1),
            prediction_head_hidden=[16],
            forward_head_hidden=[16],
        ),
    ],
    control_head=ControlHeadConfig(hidden_dims=[16, 8], output_dim=2),
    control_level_idx=1,
)
