"""Prediction and Grad-CAM shared by the demo and validation error analysis."""
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from train_baseline import ROOT, CLASSES, make_transform, plt
from model_utils import load_checkpoint
from image_framing import fit_image, crop_image


def load_run(run_name, device='cpu'):
    if not run_name or __import__('pathlib').Path(run_name).name != run_name or run_name in ('.', '..'):
        raise ValueError('Invalid run name')
    return load_checkpoint(ROOT/'outputs'/run_name/'best_model.pt', device,
                           ROOT/'data'/'split_manifest.csv')


def prepare_image(image, brightness=1., blur=0., framing='center'):
    from preview_robustness import Corrupt
    image = image.convert('RGB')
    if framing == 'center':
        # Preserve historical evaluation and Grad-CAM regression behavior.
        framed = T.Compose(make_transform(False).transforms[:2])(image)
    elif framing == 'fit':
        # Keep the full image extent and aspect ratio; pad rather than crop.
        framed = fit_image(image)
    else:
        raise ValueError(f'Unknown image framing: {framing}')
    return Corrupt(brightness, blur)(framed)


def predict_image(model, image, device='cpu', brightness=1., blur=0., framing='center'):
    prepared = prepare_image(image, brightness, blur, framing=framing)
    tensor = T.Compose(make_transform(False).transforms[2:])(prepared).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(1)[0].cpu().tolist()
    return dict(predicted_class=CLASSES[int(np.argmax(probabilities))], probabilities=probabilities), prepared


def gradcam_overlay(model, prepared_image, device='cpu', class_index=None):
    """Use an activation hook removed even on error; do not accumulate parameter grads."""
    tensor = T.Compose(make_transform(False).transforms[2:])(prepared_image).unsqueeze(0).to(device)
    captured = []
    target_layer = getattr(model, getattr(model, 'gradcam_target', 'layer4'))
    handle = target_layer.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    try:
        with torch.enable_grad():
            tensor = tensor.detach().clone().requires_grad_(True)
            logits = model(tensor)
            index = logits.argmax(1).item() if class_index is None else class_index
            activation = captured[0]
            gradient = torch.autograd.grad(logits[0, index], activation)[0]
            heat = (gradient.mean((2, 3), keepdim=True) * activation).sum(1, keepdim=True).relu()
            heat = torch.nn.functional.interpolate(heat, size=(prepared_image.height, prepared_image.width),
                                                  mode='bilinear', align_corners=False)[0, 0]
            maximum = heat.max().item()
            heat = heat.detach().cpu().numpy() / maximum if maximum > 0 else np.zeros(tuple(heat.shape))
    finally:
        handle.remove()
    colors = plt.get_cmap('jet')(heat)[..., :3]
    overlay = .55 * np.asarray(prepared_image) / 255 + .45 * colors
    return Image.fromarray(np.uint8(np.clip(overlay, 0, 1) * 255))
