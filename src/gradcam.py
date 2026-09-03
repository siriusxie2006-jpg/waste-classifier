import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    Visualisasi area gambar yang paling berkontribusi terhadap prediksi model.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model (harus dalam eval mode)
            target_layer: Layer yang akan di-hook (misal model.layer4)
        """
        self.model = model
        self.model.eval()

        self.activations = None
        self.gradients = None

        # Register forward hook untuk menangkap activations
        target_layer.register_forward_hook(self._forward_hook)
        # Register full backward hook untuk menangkap gradients
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Simpan activations dari forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Simpan gradients dari backward pass."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Index kelas target. Jika None, gunakan kelas dengan skor tertinggi.

        Returns:
            cam: Numpy array (H, W) dengan nilai 0-1, ukuran sesuai input spatial.
        """
        # Forward pass
        output = self.model(input_tensor)

        # Gunakan predicted class jika tidak dispesifikasi
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass: hitung gradient terhadap skor kelas target
        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward()

        # Global Average Pooling pada gradients → bobot per channel
        # gradients shape: (1, C, H, W) → weights shape: (C,)
        weights = self.gradients.mean(dim=(2, 3))[0]

        # Weighted combination: bobot × activations, lalu sum across channels
        # activations shape: (1, C, H, W)
        cam = (weights[:, None, None] * self.activations[0]).sum(dim=0)

        # ReLU: hanya ambil pengaruh positif
        cam = F.relu(cam)

        # Normalize ke range [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def overlay_heatmap(image_np, cam, alpha=0.5):
    """
    Overlay Grad-CAM heatmap di atas gambar asli.

    Args:
        image_np: Gambar asli dalam format numpy (H, W, 3), range [0, 255], RGB
        cam: Grad-CAM heatmap (H_cam, W_cam), range [0, 1]
        alpha: Transparansi overlay (0 = heatmap saja, 1 = gambar saja)

    Returns:
        overlaid: Gambar dengan overlay heatmap (H, W, 3), RGB, uint8
    """
    h, w = image_np.shape[:2]

    # Resize CAM ke ukuran gambar asli
    cam_resized = cv2.resize(cam, (w, h))

    # Konversi ke colormap JET (output BGR)
    heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
    # Konversi BGR → RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend gambar asli dengan heatmap
    overlaid = np.uint8(alpha * image_np + (1 - alpha) * heatmap)

    return overlaid
