import torch
from torch import nn
import torch.nn.functional as F


class APN(nn.Module):
    
    def __init__(self, in_features):
        super().__init__()

        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, 3)

        nn.init.zeros_(tensor=self.fc2.weight)
        self.fc2.bias.data.copy_(torch.tensor([0.0, 0.0, 0.5]))

    def forward(self, x):
        x = x.view(x.size(dim=0), -1)

        x = self.fc1(x)
        x = F.relu(input=x)        
        x = self.fc2(x)

        tx = x[:, 0]
        ty = x[:, 1]
        tl = x[:, 2]

        tx = torch.tanh(tx)
        ty = torch.tanh(ty)
        tl = torch.sigmoid(tl)

        return tx, ty, tl


def crop_image(image, tx, ty, tl):
    b, _, _, _ = image.size()

    theta = torch.zeros(size=(b, 2, 3), dtype=image.dtype, device=image.device)
    theta[:, 0, 0] = tl
    theta[:, 0, 2] = tx
    theta[:, 1, 1] = tl
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta=theta, size=image.size(), align_corners=False)
    cropped_image = F.grid_sample(input=image, grid=grid, align_corners=False)

    return cropped_image


def rank_loss(logits_coarse, logits_fine, labels, margin=0.05) -> torch.Tensor:
    """랭크 손실 함수"""    
    probabillity_coarse = F.softmax(input=logits_coarse, dim=1)
    probabillity_fine = F.softmax(input=logits_fine, dim=1)

    batch_size = labels.size(dim=0)
    batch_indices = torch.arange(end=batch_size, device=labels.device)

    probabillity_coarse_correct = probabillity_coarse[batch_indices, labels]
    probabillity_fine_correct = probabillity_fine[batch_indices, labels]

    target = torch.ones(size=[batch_size], device=labels.device)

    loss = F.margin_ranking_loss(
        input1=probabillity_fine_correct,
        input2=probabillity_coarse_correct,
        target=target,
        margin=margin,
    )

    return loss