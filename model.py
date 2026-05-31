import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class LightweightBackbone(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.layer4 = self._make_layer(out_channels, 2, stride=2)

    def _make_layer(self, out_channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ImageQualityGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        hidden_channels = max(in_channels // 8, 1)
        self.quality_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scores = self.quality_net(x).squeeze(-1).squeeze(-1)
        return {
            "clarity": scores[:, 0],
            "occlusion": scores[:, 1],
            "illumination": scores[:, 2]
        }


class EdgeEnhancement(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        edge_feat = self.conv(x)
        attention = self.sigmoid(self.bn(edge_feat))
        return x * attention


class ContextCompensation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        y = self.gap(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y


class DynamicFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, f_orig, f_edge, f_context, quality_scores):
        weights = self.fc(quality_scores)
        fused = (
            weights[:, 0:1, None, None] * f_orig
            + weights[:, 1:2, None, None] * f_edge
            + weights[:, 2:3, None, None] * f_context
        )
        return fused


class DynamicMeterRecognitionModel(nn.Module):
    def __init__(self, num_digits=5, num_classes=10, backbone_channels=256):
        super().__init__()
        self.num_digits = num_digits
        self.num_classes = num_classes
        self.backbone = LightweightBackbone(out_channels=backbone_channels)
        self.quality_gate = ImageQualityGate(backbone_channels)
        self.edge_enhance = EdgeEnhancement(backbone_channels)
        self.context_comp = ContextCompensation(backbone_channels)
        self.dynamic_fusion = DynamicFusion(backbone_channels)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.digit_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_channels, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
            for _ in range(num_digits)
        ])

    def forward(self, x):
        features = self.backbone(x)
        quality_info = self.quality_gate(features)
        f_orig = features
        f_edge = self.edge_enhance(features)
        f_context = self.context_comp(features)
        quality_scores = torch.stack([
            quality_info["clarity"],
            quality_info["occlusion"],
            quality_info["illumination"]
        ], dim=1)
        fused = self.dynamic_fusion(f_orig, f_edge, f_context, quality_scores)
        pooled = self.adaptive_pool(fused)
        flattened = pooled.view(pooled.size(0), -1)
        digit_predictions = [head(flattened) for head in self.digit_heads]
        predictions = torch.stack(digit_predictions, dim=1)
        return predictions, {"quality_info": quality_info}


def create_model(num_digits=5, num_classes=10, device=None, backbone_channels=256):
    model = DynamicMeterRecognitionModel(
        num_digits=num_digits,
        num_classes=num_classes,
        backbone_channels=backbone_channels
    )
    if device is not None:
        model = model.to(device)
    return model
