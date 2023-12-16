from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from gym import spaces
import torch
class Customlstm(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self,observation_space: spaces.Dict, features_dim: int = 1):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.L = [0.5, 0.5, 0.5, 0.5, 0.5]
        self.lamda = []
        for i in range(1, observation_space.spaces["elem"].shape[0]):
            self.lamda.append([j ** i for j in self.L])
        self.lamda = torch.Tensor(self.lamda).unsqueeze(dim=0)


        self.linear_0 = nn.Linear(14, 512)
        self.activation_1 = nn.GELU()
        self.linear_1 = nn.Linear(512, 512)
        self.activation_2 = nn.GELU()
        self.linear_2 = nn.Linear(512, 512)
        self.activation_3 = nn.GELU()

    def forward(self, input):
        Diction = [input.get(key) for key in ["temp_new", "humid_new", "elem", "humider", "fan"]]
        x = torch.stack(Diction, -2).permute(0, 2, 1)
        reminder = torch.cat([input.get(key) for key in ["temp_fix", "temp_out", "humid_fix", "humid_out"]], 1)
        reminder = torch.cat([reminder, x[:, -1, :]], 1)
        x = (x[:, :-1, :]*self.lamda.cuda()).sum(dim=1)


        x = torch.cat((reminder.cuda(), x,), 1)
        x = self.linear_0(x)
        x = self.activation_1(x)
        x = self.linear_1(x)
        x = self.activation_2(x)
        x = self.linear_2(x)
        x = self.activation_3(x)

        return x

