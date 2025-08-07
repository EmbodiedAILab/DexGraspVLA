import abc
from typing import Dict

from controller.policy.base_image_policy import BaseImagePolicy
from serving.base_policy import BasePolicy


class DexGraspVLAPolicy(BasePolicy):
    def __init__(self, policy: BaseImagePolicy):
        super().__init__()
        self.policy = policy

    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""
        return self.policy.predict_action(obs)


    def reset(self) -> None:
        """Reset the policy to its initial state."""
        self.policy.reset()
