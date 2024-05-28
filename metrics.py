import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        targets = targets.float()

        numerator = torch.sum(inputs * targets)
        denominator = torch.sum(inputs) + torch.sum(targets)

        dice = (2. * numerator + self.smooth) / (denominator + self.smooth)

        return 1. - dice


class CombinationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        return 0.5 * self.dice(inputs, targets) + 0.5 * self.bce(inputs, targets)

