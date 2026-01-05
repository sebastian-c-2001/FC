import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


# Arhitectura VoxelMorph simplificată (U-Net + Spatial Transformer)
class VoxelMorphInference(nn.Module):
    def __init__(self, size=(160, 192)):
        super().__init__()
        # Definirea grilei pentru deformare
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids).unsqueeze(0).float()
        self.register_buffer('grid', grid)

        # Encoder/Decoder simplu (Specific VoxelMorph)
        self.unet = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1)  # Produce flow-ul (u)
        )

    def forward(self, moving, fixed):
        x = torch.cat([moving, fixed], dim=1)
        flow = self.unet(x)

        # Aplicarea transformării (Warping)
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]
        warped = F.grid_sample(moving, new_locs, align_corners=True)
        return warped, flow


# Exemplu de utilizare
def run_registration(img_path_m, img_path_f, model_weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VoxelMorphInference().to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()

    # Pre-procesare imagini
    m = torch.from_numpy(np.array(Image.open(img_path_m))).float().unsqueeze(0).unsqueeze(0) / 255.0
    f = torch.from_numpy(np.array(Image.open(img_path_f))).float().unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        warped, flow = model(m.to(device), f.to(device))
    return warped.squeeze().cpu().numpy()