
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.models import create_model
from DualPathModel import DualPath_Baseline, DualPath_Fusion
from PIL import Image

# Global hook containers
features = []
gradients = []

def save_gradcam(img_tensor, cam, save_path):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)

    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    plt.imsave(save_path, overlay)

def extract_feature_hook(module, input, output):
    features.append(output)

def extract_gradient_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def generate_gradcam(model, image_tensor, class_idx, target_layer, device):
    features.clear()
    gradients.clear()

    handle_fwd = target_layer.register_forward_hook(extract_feature_hook)
    handle_bwd = target_layer.register_full_backward_hook(extract_gradient_hook)

    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device).requires_grad_()
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()

    loss = output[0, class_idx]
    loss.backward()

    grads_val = gradients[0].squeeze(0)
    fmap = features[0].squeeze(0)

    weights = grads_val.mean(dim=(1, 2))
    cam = torch.sum(weights[:, None, None] * fmap, dim=0)
    cam = F.relu(cam)

    handle_fwd.remove()
    handle_bwd.remove()

    return cam.detach(), pred_class

def main():
    model_name = 'baseline'  # or 'fusion'
    dataset_path = '../dataset/RAF-DB/test'
    checkpoint_path = './logs/v1/base_ddga_rafdb/baseline_RAF-DB_best.pt'
    save_dir = 'evaluation_logs/gradcam'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'baseline':
        model = DualPath_Baseline(num_classes=7)
    else:
        model = DualPath_Fusion(num_classes=7)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    model.to(device)

    # Ambil layer backbone terakhir
    target_layer = model.backbone.blocks[-1]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(dataset_path, transform=transform)

    idxs = random.sample(range(len(dataset)), 10)
    for i, idx in enumerate(idxs):
        image_tensor, label = dataset[idx]
        cam, pred = generate_gradcam(model, image_tensor, label, target_layer, device)
        save_path = os.path.join(save_dir, f"gradcam_{i}_pred{pred}_true{label}.png")
        save_gradcam(image_tensor, cam, save_path)
        print(f"[✓] Saved: {save_path}")

if __name__ == '__main__':
    main()
