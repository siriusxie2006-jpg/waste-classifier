import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    CNN sederhana dengan 3 block konvolusi untuk baseline klasifikasi sampah.
    Input: 224x224x3 RGB image
    Output: num_classes logits
    """

    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()

        # 3 Convolutional blocks: Conv2d -> ReLU -> MaxPool2d
        # Setiap block mengurangi spatial dimension 1/2 dan menambah channel
        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels, 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32 -> 64 channels, 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 -> 128 channels, 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected classifier
        # Input: 128 channels * 28 * 28 = 100352
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),          # Dropout untuk mencegah overfitting
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
