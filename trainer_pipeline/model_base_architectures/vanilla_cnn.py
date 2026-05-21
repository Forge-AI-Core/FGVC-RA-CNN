import torch.nn as nn
from torchvision.models import vgg19_bn, VGG19_BN_Weights


class VanillaCNN(nn.Module):
    def __init__(self, num_classes: int = 200):
        super().__init__()
        
        base = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)        
        self.features = base.features        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)), 
            nn.Flatten(),              
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),                  
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        
        return logits

