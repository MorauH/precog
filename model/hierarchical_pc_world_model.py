import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .fnn import FNN
from .ssm import SelectiveSSM
from .pc_level import PCLevel
from .modality_encoder import ModalityEncoder
from .control_head import ControlHead
from .config import ModelConfig


class HierarchicalPCWorldModel(nn.Module):
    """
    Full Hierarchical Predictive Coding World Model with JEPA stabilization.
    
    Architecture:
      - Level 0: Per-modality encoders (preserve sensor identity)
      - Level 1+: PCLevels (each contains SelectiveSSM + prediction heads)
      - Control head on top of mid-level (z_mid + predicted z_mid_next)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Level 0: Modality-specific encoders
        self.modality_encoders: nn.ModuleDict = nn.ModuleDict()
        for mod in config.modalities:
            self.modality_encoders[mod.name] = ModalityEncoder(
                input_dim=mod.input_dim,
                output_dim=mod.output_dim,
                hidden_dim=mod.hidden_dims if hasattr(mod, 'hidden_dims') else None
            )
        
        # PC Levels
        self.levels: nn.ModuleList[PCLevel] = nn.ModuleList()
        prev_dim = config.d_level0
        
        for i, level_cfg in enumerate(config.level_configs):
            self.levels.append(PCLevel(
                input_dim=prev_dim,                    # d_below
                repr_dim=level_cfg.d_representation,   # d_representation
                state_dim=level_cfg.ssm.d_state,
                prediction_head_hidden=level_cfg.prediction_head_hidden,
                forward_head_hidden=level_cfg.forward_head_hidden,
                dt_min=level_cfg.ssm.dt_min,
                dt_max=level_cfg.ssm.dt_max,
                name=f"level{i+1}"
            ))
            prev_dim = level_cfg.d_representation   # next level receives this repr dim
        
        # Control Head
        ctrl_cfg = config.control_head
        control_input_dim = config.level_configs[config.control_level_idx].d_representation * 2  # z + z_hat
        
        self.control_head = ControlHead(
            input_dim=control_input_dim,
            hidden_dims=ctrl_cfg.hidden_dims,
            output_dim=ctrl_cfg.output_dim
        )
        
        self.control_level_idx = config.control_level_idx
        
        # Hidden states for online inference
        self.reset_hidden_states(batch_size=1)

    def reset_hidden_states(self, batch_size: int = 1):
        """Reset all hidden states (for new episode or online reset)."""
        self.hidden_states = []
        self.hidden_states_target = []
        
        for level in self.levels:
            h, h_target = level.init_hidden(batch_size)
            self.hidden_states.append(h)
            self.hidden_states_target.append(h_target)
    
   def forward(self, 
                obs_dict: Dict[str, torch.Tensor],
                prev_action: Optional[torch.Tensor] = None,
                return_all: bool = False):
        
        batch, seq_len = next(iter(obs_dict.values())).shape[:2]
        
        if self.hidden_states[0].shape[0] != batch:
            self.reset_hidden_states(batch)
        
        # Level 0 encoding
        level0_list = []
        for name, encoder in self.modality_encoders.items():
            x = obs_dict[name]
            if name == "control" and prev_action is not None:
                x = prev_action.unsqueeze(1) if x.dim() == 2 else prev_action
            z0 = encoder(x)
            level0_list.append(z0)
        
        z_level = torch.cat(level0_list, dim=-1)   # (B, T, d_level0)
        
        # Hierarchical forward
        level_outputs = []
        z_hat_next_list = []
        prediction_errors = []
        
        for i, level in enumerate(self.levels):
            pred_from_above = level_outputs[-1] if i > 0 else None
            
            z_seq, z_hat_next_seq, pred_below_seq, h_new, h_target_new, epsilon = level.forward(
                z_level, 
                self.hidden_states[i],
                self.hidden_states_target[i],
                pred_from_above
            )
            
            # Keep last hidden state for online stepping
            self.hidden_states[i] = h_new[:, -1]
            self.hidden_states_target[i] = h_target_new[:, -1]
            
            level_outputs.append(z_seq)
            z_hat_next_list.append(z_hat_next_seq)
            prediction_errors.append(epsilon)
            
            z_level = z_seq  # feed to next level
        
        # Control head from chosen level
        ctrl_idx = self.control_level_idx
        z_ctrl = level_outputs[ctrl_idx]
        z_hat_ctrl = z_hat_next_list[ctrl_idx]
        
        action_pred = self.control_head(torch.cat([z_ctrl, z_hat_ctrl], dim=-1))
        
        if not return_all:
            return action_pred
        
        return {
            'action_pred': action_pred,
            'z_levels': level_outputs,
            'z_hat_next': z_hat_next_list,
            'prediction_errors': prediction_errors,
            'total_surprise': sum(e.pow(2).mean() for e in prediction_errors)
        }
   
   def step(self, obs_dict: Dict[str, torch.Tensor], prev_action: torch.Tensor) -> Dict:
       """Single timestep online step."""
       result = self.forward(
           {k: v.unsqueeze(1) for k, v in obs_dict.items()},
           prev_action,
           return_all=True
       )
       
       return {
           'action': result['action_pred'][:, -1],
           'z_mid': result['z_levels'][self.control_level_idx][:, -1],
           'prediction_errors': result['prediction_errors'],
           'surprise': result['total_surprise']
       }

   def update_ema(self, tau: float = 0.997):
       """Update all target encoders (call after optimizer.step())."""
       for level in self.levels:
           level.update_ema(tau)
