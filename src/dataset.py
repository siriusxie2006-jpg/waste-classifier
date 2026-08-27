import os
from PIL import Image
from torch.utils.data import Dataset


class WasteDataset(Dataset):
    """
    Custom PyTorch Dataset untuk klasifikasi sampah.
    Membaca gambar dari struktur folder: root_dir/nama_kelas/gambar.jpg
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path ke folder dataset (misal: data/train)
            transform: Transformasi torchvision yang akan diterapkan ke gambar
        """
        self.root_dir = root_dir
        self.transform = transform

        # Buat mapping kelas -> index secara alphabetical
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Kumpulkan semua path gambar beserta labelnya
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Buka gambar dan pastikan format RGB (beberapa gambar bisa grayscale/RGBA)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label