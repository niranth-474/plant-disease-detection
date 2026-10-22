
import torch
import torch.nn as nn


# CNN building block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            # Convolution 1
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Convolution 2
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Convolution 3
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Reduce spatial dimensions
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)


# Custom Plant Disease CNN
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # CNN feature extractor
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )

        # Convert each feature map into one value
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ANN classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():

            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight,
                    nonlinearity="relu"
                )
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # CNN: learn spatial features from the leaf
        x = self.features(x)

        # Pooling: summarize each feature map
        x = self.global_pool(x)

        # ANN: classify the extracted features
        x = self.classifier(x)

        # Return raw class scores (logits)
        return x


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = PlantDiseaseCNN(num_classes=38).to(device)

    sample = torch.randn(4, 3, 224, 224).to(device)

    model.eval()
    with torch.no_grad():
        output = model(sample)

    total_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("Device:", device)
    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print(f"Trainable parameters: {total_params:,}")
