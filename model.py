import torch
from torch import nn


class DrawingCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


if __name__ == "__main__":
    model = DrawingCNN()
    model.eval()

    sample_image = torch.zeros(1, 1, 28, 28)

    with torch.no_grad():
        outputs = model(sample_image)

    print("Input shape:", tuple(sample_image.shape))
    print("Output shape:", tuple(outputs.shape))
    print("Model structure check passed.")