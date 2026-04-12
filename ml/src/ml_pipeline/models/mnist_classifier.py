import torch.nn as nn


class MNISTClassifier(nn.Module):

    def __init__(self, model_params):
        super().__init__()
        self.model_params = model_params

        self.model = nn.Sequential(
            ...
        )

