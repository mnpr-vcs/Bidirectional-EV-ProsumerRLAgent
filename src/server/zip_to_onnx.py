from typing import Tuple
import onnx
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.policies import BasePolicy


# for ppo (onpolicy)
class OnnxableOnPolicy(torch.nn.Module):
    def __init__(self, policy: BasePolicy):
        """
        Initializes the OnnxableOnPolicy class.
        Parameters:
            policy (BasePolicy): The policy to be used.
        """
        super().__init__()
        self.policy = policy  # policy network

    def forward(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward method for the OnnxableOnPolicy class.

        Parameters:
            observation (torch.Tensor): The input observation tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing three torch tensors.
            [action, log_prob, value]
        """
        action, _, _ = self.policy.forward(observation)
        return torch.clamp(action, -1, 1)


# for sac, td3 (offpolicy)
class OnnxableOffPolicy(torch.nn.Module):
    def __init__(self, actor: torch.nn.Module):
        """
        Initializes the OnnxableOffPolicy class.
        Parameters:
            actor (torch.nn.Module): The actor module.
        """
        super().__init__()
        self.actor = actor  # actor network

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            observation (torch.Tensor): The input observation tensor.
        Returns:
            torch.Tensor: action.
        """
        action = self.actor(observation, deterministic=True)
        return torch.clamp(action, -1, 1)


if __name__ == "__main__":

    model_ppo = PPO.load("../results/checkpoints/ppo.zip", device="cpu")
    model_sac = SAC.load("../results/checkpoints/sac.zip", device="cpu")
    model_td3 = TD3.load("../results/checkpoints/td3.zip", device="cpu")

    onnx_policy_ppo = OnnxableOnPolicy(model_ppo.policy)
    onnx_policy_sac = OnnxableOffPolicy(model_sac.policy)
    onnx_policy_td3 = OnnxableOffPolicy(model_td3.policy)

    observation_size = model_ppo.observation_space.shape
    dummy_input = torch.randn(1, *observation_size)
    print(f">> input shape: {dummy_input.shape}")
    torch.onnx.export(
        onnx_policy_ppo,
        dummy_input,
        "checkpoints/ppo_model.onnx",
        opset_version=17,
        input_names=["observations"],
    )
    torch.onnx.export(
        onnx_policy_sac,
        dummy_input,
        "checkpoints/sac_model.onnx",
        opset_version=17,
        input_names=["observations"],
    )
    torch.onnx.export(
        onnx_policy_td3,
        dummy_input,
        "checkpoints/td3_model.onnx",
        opset_version=17,
        input_names=["observations"],
    )
    print(">> Done !!")
