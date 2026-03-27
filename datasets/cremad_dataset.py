import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
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

    # def __init__(self, mode, cfg): # Previous version
    def __init__(self, mode, config=None, **kwargs):
        """
        Initializes the CREMAD dataset.

        Args:
            mode (str): Dataset mode ('train', 'val', or 'test').
            config (object, optional): Configuration object (e.g., OmegaConf DictConfig)
                                       whose attributes override CREMADConfig defaults.
            **kwargs: Additional keyword arguments to override CREMADConfig defaults.
        """
        # 1. Initialize base config
        self.config = CREMADConfig()

        # 2. Override with hydra/main config if provided
        if config is not None:
            # Convert OmegaConf to dict or handle attribute access
            # Assuming config is DictConfig or similar
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # 3. Override with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        # self.cfg = cfg # No longer needed, use self.config

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
                # Use config for FPS in visual path
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

        # --- Audio Processing (Original Logic with Config Params) ---
        target_sr = self.config.audio_sr
        target_len_secs = self.config.audio_duration_secs # Original used tiling to 3 seconds
        target_len_samples = target_sr * target_len_secs

        samples, rate = librosa.load(self.audio[idx], sr=target_sr)

        # Original tiling logic: tile to 3 times original length, then take first 3 seconds
        # This ensures output length is consistent even for short files.
        # Using explicit target length derived from config is clearer.
        if len(samples) == 0: # Handle empty audio files
            print(f"Warning: Empty audio file encountered: {self.audio[idx]}")
            samples = np.zeros(target_len_samples, dtype=np.float32)
        else:
            # Tile samples to be at least target_len_samples long
            n_repeat = int(np.ceil(target_len_samples / len(samples)))
            samples = np.tile(samples, n_repeat)
            # Truncate to the exact target length
            samples = samples[:target_len_samples]

        # Clamp values (as in original code)
        samples[samples > 1.] = 1.
        samples[samples < -1.] = -1.

        # Spectrogram using config parameters
        spectrogram = librosa.stft(samples,
                                   n_fft=self.config.audio_n_fft,
                                   hop_length=self.config.audio_hop_length)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        # --- Visual Processing (Original Logic with Config Params) ---
        img_size = self.config.visual_image_size
        norm_mean = self.config.visual_norm_mean
        norm_std = self.config.visual_norm_std

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])
        else: # val or test
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])

        image_dir = self.image[idx]
        try:
            image_samples = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        except FileNotFoundError:
             raise FileNotFoundError(f"Image directory not found: {image_dir}")

        if not image_samples:
             raise ValueError(f"No image frames found in {image_dir}")

        # Original logic: Select exactly 1 random frame
        select_index = np.random.choice(len(image_samples), size=1, replace=False)
        # No need to sort select_index as it has only one element

        # Original logic: create tensor for 1 frame (T=1)
        # Shape (1, 3, H, W)
        images_tensor = torch.zeros((1, 3, img_size, img_size))

        # Original logic: loop exactly once
        img_path = os.path.join(image_dir, image_samples[select_index[0]])
        try:
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images_tensor[0] = img # Assign to the first (and only) frame index
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Handle error, e.g., use a placeholder? Using zeros.
            images_tensor[0] = torch.zeros((3, img_size, img_size))

        # Original permutation: (1, 3, H, W) -> (3, 1, H, W)
        images_tensor = images_tensor.permute(1, 0, 2, 3)

        # --- Label ---
        label = self.label[idx]

        # --- Output Dict (Original Structure) ---
        output = {
            # Original audio shape: (F, T) -> unsqueeze(1) -> (F, 1, T) -> permute(1, 0, 2) -> (1, F, T)
            # Current audio shape: (F, T) -> unsqueeze(0) -> (1, F, T)
            # The current shape is correct and matches common conventions (Batch, Channel, Freq, Time)
            # Keeping the simpler unsqueeze(0) from the previous refactor.
            'audio': torch.from_numpy(spectrogram).unsqueeze(0).float(),
            'clip': images_tensor.float(), # Use the tensor processed with original logic
            'target': label,
            'index' : idx
        }

        return output