from torch import nn
from torch.optim import Adam
from torchvision import models

#create a class for an imported resnet model from torch. Replce the last layer with a new layer with the number of classes
class ResNetModel(nn.Module):
    def __init__(self, num_classes: int, freeze_features: bool = True):
        super(ResNetModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        if freeze_features:
            for param in self.model.parameters():
                param.required_grad = False
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
                                nn.Linear(in_features, 256),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(256, num_classes)
                            )

    def forward(self, x):
        return self.model(x)
    
class VGGModel(nn.Module):
    def __init__(self, num_classes: int, freeze_features: bool = True):
        super(VGGModel, self).__init__()
        self.model = models.vgg16(pretrained=True)
        if freeze_features:
            for param in self.model.features.parameters():
                param.required_grad = False
        in_features = self.model.classifier[-1].in_features
        features = list(self.model.classifier.children())[:-1] 
        features.extend([nn.Sequential(
                                nn.Linear(in_features, 256), 
                                nn.ReLU(),
                                nn.Dropout(0.3), 
                                nn.Linear(256, num_classes)
                                )
                        ]
        )
        self.model.classifier = nn.Sequential(*features)

    def forward(self, x):
        return self.model(x)