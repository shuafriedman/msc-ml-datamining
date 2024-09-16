from torch import nn
from torchvision import models
import timm

class ResNetModel(nn.Module):
    def __init__(self, num_classes: int, param_size: int = 256, freeze_features: bool = True):
        super(ResNetModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.config = None

        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(in_features, param_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(param_size, num_classes)
        )

        for m in self.model.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

class VGGModel(nn.Module):
    def __init__(self, num_classes: int, param_size: int = 256, freeze_features: bool = True):
        super(VGGModel, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.config = None

        if freeze_features:
            for param in self.model.features.parameters():
                param.requires_grad = False

        in_features = self.model.classifier[-1].in_features

        features = list(self.model.classifier.children())[:-1] 
        features.extend([nn.Sequential(
            nn.Linear(in_features, param_size), 
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(param_size, num_classes)
        )])

        self.model.classifier = nn.Sequential(*features)

        for m in self.model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

class Eva(nn.Module):
    def __init__(self, num_classes: int, param_size: int = 256, freeze_features: bool = True):
        super(Eva, self).__init__()
        self.model = timm.create_model('eva_large_patch14_336.in22k_ft_in22k_in1k', pretrained=True)
        self.config = timm.data.resolve_model_data_config(self.model)

        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.head.in_features

        self.model.head = nn.Sequential(
            nn.Linear(in_features, param_size), 
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(param_size, num_classes)
        )

        for m in self.model.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)
