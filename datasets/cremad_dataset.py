import csv
import os
import librosa
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class CREMADConfig:
    """Configuration settings for the CREMAD dataset."""
    # Paths and file names (defaults can be overridden by main config)
    name: str = "CREMAD"
    data_root: str = "./CREMAD/"
    train_list: str = "train.csv"
    test_list: str = "test.csv"
    visual_feature_path: str = "./CREMAD" # Default base path
    audio_feature_path: str = "./CREMAD/AudioWAV" # Default

    # Class settings
    n_classes: int = 6
    class_dict: Dict[str, int] = field(default_factory=lambda: {
        'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5
    })

    # Audio settings
    audio_sr: int = 22050
    audio_n_fft: int = 512
    audio_hop_length: int = 353
    audio_duration_secs: int = 3 # Original code tiled to 3 seconds
    # audio_norm_mean_std: bool = False # Example: if normalization needed

    # Visual settings
    visual_fps: int = 1 # FPS used in the directory structure
    visual_image_size: int = 224
    visual_num_frames: int = 1 # Number of frames to sample per video
    visual_norm_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    visual_norm_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Other
    embed_size: Optional[int] = None # Example if needed


class CREMADDataset(Dataset):

    def __init__(self, mode, config=None, **kwargs):
        """
        Initializes the CREMAD dataset.

        Args:
            mode (str): Dataset mode ('train', 'val', or 'test').
            config (object, optional): Configuration object (e.g., OmegaConf DictConfig)
                                       whose attributes override CREMADConfig defaults.
            **kwargs: Additional keyword arguments to override CREMADConfig defaults.
        """
        self.config = CREMADConfig()

        if config is not None:
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        img_size = self.config.visual_image_size
        normalization = transforms.Normalize(
            self.config.visual_norm_mean,
            self.config.visual_norm_std,
        )
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalization,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalization,
            ])

        self.data_root = self.config.data_root
        class_dict = self.config.class_dict

        self.visual_feature_base_path = self.config.visual_feature_path
        self.audio_feature_path = self.config.audio_feature_path

        self.train_csv = os.path.join(self.data_root, self.config.train_list)
        self.test_csv = os.path.join(self.data_root, self.config.test_list)

        if mode == 'train':
            csv_file = self.train_csv
        elif mode in ['val', 'test']:
            csv_file = self.test_csv
        else:
             raise ValueError(f"Invalid mode: {mode}. Expected 'train', 'val', or 'test'.")

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                if len(item) < 2:
                    print(f"Skipping malformed line: {item}")
                    continue

                file_id = item[0]
                label_str = item[1].strip()

                audio_path = os.path.join(self.audio_feature_path, file_id + '.wav')
                visual_subdir = 'Image-05-FPS'
                visual_path = os.path.join(self.visual_feature_base_path, visual_subdir, file_id)
                
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    if label_str in class_dict:
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[label_str])
                    else:
                        print(f"Warning: Unknown label '{label_str}' in {csv_file} for item {file_id}. Skipping.")
                        continue
                else:
                    if not os.path.exists(audio_path):
                        print(f"Audio Path not found: {audio_path}")
                    if not os.path.exists(visual_path):
                        print(f"Visual Path not found: {visual_path}")
                    continue

        print(f'{self.mode} data load finish')
        print(f'# of files = {len(self.image)} ')

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        target_sr = self.config.audio_sr
        target_len_secs = self.config.audio_duration_secs
        target_len_samples = target_sr * target_len_secs

        samples, _ = librosa.load(self.audio[idx], sr=target_sr)

        if len(samples) == 0:
            print(f"Warning: Empty audio file encountered: {self.audio[idx]}")
            samples = np.zeros(target_len_samples, dtype=np.float32)
        else:
            n_repeat = int(np.ceil(target_len_samples / len(samples)))
            samples = np.tile(samples, n_repeat)
            samples = samples[:target_len_samples]

        samples[samples > 1.] = 1.
        samples[samples < -1.] = -1.

        spectrogram = librosa.stft(samples,
                                   n_fft=self.config.audio_n_fft,
                                   hop_length=self.config.audio_hop_length)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        img_size = self.config.visual_image_size

        image_dir = self.image[idx]
        try:
            image_samples = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        except FileNotFoundError:
             raise FileNotFoundError(f"Image directory not found: {image_dir}")

        if not image_samples:
             raise ValueError(f"No image frames found in {image_dir}")

        select_index = np.random.choice(len(image_samples), size=1, replace=False)

        images_tensor = torch.zeros((1, 3, img_size, img_size))

        img_path = os.path.join(image_dir, image_samples[select_index[0]])
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            images_tensor[0] = img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            images_tensor[0] = torch.zeros((3, img_size, img_size))

        images_tensor = images_tensor.permute(1, 0, 2, 3)

        label = self.label[idx]

        output = {
            'audio': torch.from_numpy(spectrogram).unsqueeze(0).float(),
            'clip': images_tensor.float(),
            'target': label,
            'index' : idx
        }

        return output
