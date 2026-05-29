from dataclasses import dataclass, field
from typing import List


@dataclass
class ModalityConfig:
    """
    Configuration for a single sensor modality encoder (Level 0).

    Attributes:
        name:       Human-readable identifier, e.g. "imu", "lidar", "speed"
        input_dim:  Raw feature dimension coming from the existing pipeline
        output_dim: Projected latent dimension fed into Level 1
        hidden_dims: Optional hidden layers in the encoder FNN.
                     Empty list = single linear projection (recommended for
                     low-dim modalities like speed/control).
    """
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: List[int] = field(default_factory=list)


@dataclass
class SSMConfig:
    """
    Configuration for one SelectiveSSM inside a PC level.

    Attributes:
        d_state:   Internal SSM state dimension (memory capacity)
        dt_min:    Minimum timescale Δ (controls fastest dynamics)
        dt_max:    Maximum timescale Δ (controls slowest dynamics)

    Note: d_input and d_output are inferred from PCLevelConfig.
    """
    d_state: int
    dt_min: float = 0.001
    dt_max: float = 0.1


@dataclass
class PCLevelConfig:
    """
    Configuration for one level of the PC hierarchy.

    Attributes:
        d_representation:    Output dimension of this level's SSM.
                             Also the dimension of top-down predictions
                             this level receives from the level above.
        ssm:                 SSM configuration for this level.
        prediction_head_hidden: Hidden dims of the FNN that predicts
                                the level below's representation.
        forward_head_hidden: Hidden dims of the FNN that predicts this
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
        modalities:     List of Level 0 modality encoder configs.
                        The concatenated output dims must equal
                        level_configs[0]'s implied d_below.
        level_configs:  PC hierarchy levels, ordered low (fast) to high (slow).
                        Exactly 3 levels expected (low / mid / high).
        control_head:   Control output head config.
        control_level_idx: Which PC level index feeds the control head.
                           Default 1 = mid-level (0-indexed).
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
    def d_level0(self) -> int:
        """Total dimension of concatenated Level 0 representations."""
        return sum(m.output_dim for m in self.modalities)


# ---------------------------------------------------------------------------
# Default config
# Sized for a Jetson-class embedded computer and a few hours of driving logs.
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = ModelConfig(
    modalities=[
        ModalityConfig(name="imu",     input_dim=6,  output_dim=16, hidden_dims=[]),
        ModalityConfig(name="lidar",   input_dim=32, output_dim=32, hidden_dims=[32]),
        ModalityConfig(name="speed",   input_dim=4,  output_dim=8,  hidden_dims=[]),
        ModalityConfig(name="control", input_dim=2,  output_dim=8,  hidden_dims=[]),
    ],
    level_configs=[
        # Level 1 — fast, reactive dynamics  (d_below = 64 = sum of modality dims)
        PCLevelConfig(
            d_representation=128,
            ssm=SSMConfig(d_state=64, dt_min=0.001, dt_max=0.01),
            prediction_head_hidden=[64],
            forward_head_hidden=[64],
        ),
        # Level 2 — mid, dynamic state  (d_below = 128)
        PCLevelConfig(
            d_representation=128,
            ssm=SSMConfig(d_state=64, dt_min=0.01, dt_max=0.1),
            prediction_head_hidden=[64],
            forward_head_hidden=[64],
        ),
        # Level 3 — slow, situational context  (d_below = 128)
        PCLevelConfig(
            d_representation=64,
            ssm=SSMConfig(d_state=32, dt_min=0.1, dt_max=1.0),
            prediction_head_hidden=[64],
            forward_head_hidden=[32],
        ),
    ],
    control_head=ControlHeadConfig(hidden_dims=[64, 32], output_dim=2),
    control_level_idx=1,
)
