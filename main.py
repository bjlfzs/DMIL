import hydra
import torch
from utils import setup_seed, cross_modality_pretrain
from train import train_model


def load_pretrained_resnet(video_model, audio_model, checkpoint_path, channel=1):
    """Load pretrained ResNet weights and adapt the audio encoder's input channel.

    Args:
        video_model: Visual encoder (ResNet).
        audio_model: Audio encoder (ResNet, adapted from RGB weights).
        checkpoint_path: Path to the pretrained ResNet checkpoint (.pth).
        channel: Number of input channels for the audio encoder (default: 1 for spectrogram).

    Returns:
        Tuple of (video_model, audio_model) with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint.pop('fc.weight')
    checkpoint.pop('fc.bias')
    video_model.load_state_dict(checkpoint)
    checkpoint['conv1.weight'] = cross_modality_pretrain(checkpoint['conv1.weight'], channel)
    audio_model.load_state_dict(checkpoint)
    return video_model, audio_model


def build_model(cfg):
    """Build and return the DMILModel with pretrained ResNet encoders.

    Args:
        cfg: Hydra config object containing encoder, dataset, and method configs.

    Returns:
        DMILModel instance (possibly wrapped in CustomDataParallel).
    """
    from models.multimodal import DMILModel, CustomDataParallel

    pretrained_path = cfg.get('pretrained_path', './pretrained/resnet18-f37072fd.pth')

    # Resolve the target device before instantiating any modules.
    if cfg.device != "cpu":
        cfg.device = "cuda:" + str(cfg.gpu_device[0])

    video_encoder = hydra.utils.instantiate(cfg.encoder_v).to(cfg.device)
    audio_encoder = hydra.utils.instantiate(cfg.encoder_a).to(cfg.device)
    video_encoder, audio_encoder = load_pretrained_resnet(
        video_encoder, audio_encoder, pretrained_path, channel=1
    )

    model = DMILModel(
        video_encoder, audio_encoder,
        cfg.embed_size, cfg.n_classes, cfg.methods
    )

    if cfg.device != "cpu" and len(cfg.gpu_device) > 1:
        model = CustomDataParallel(model, device_ids=cfg.gpu_device).to(cfg.device)
        print(f"Using CustomDataParallel on devices {cfg.gpu_device}")
    else:
        model = model.to(cfg.device)
        print(f"Using device {cfg.device}")

    return model


@hydra.main(config_path='cfgs', config_name='train', version_base=None)
def main(cfg):
    """Main entry point."""
    # Initialize random seed for reproducibility
    setup_seed(cfg.random_seed)

    # Build model
    model = build_model(cfg)

    if cfg.train:
        # Train model
        train_results = train_model(cfg, model)
        print("\nTraining and evaluation complete.")


if __name__ == '__main__':
    main()
