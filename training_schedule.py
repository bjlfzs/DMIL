"""Pure helpers for the paper's three-stage DMIL training schedule."""


def validate_schedule(stage1_epochs, stage2_epochs, n_epochs):
    """Validate epoch counts before the expensive training setup starts."""
    values = {
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "n_epochs": n_epochs,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, got {value!r}")

    if stage1_epochs <= 0:
        raise ValueError("stage1_epochs must be greater than zero")
    if stage2_epochs < 0:
        raise ValueError("stage2_epochs must be non-negative")
    if n_epochs <= 0:
        raise ValueError("n_epochs must be greater than zero")
    if n_epochs < stage1_epochs:
        raise ValueError(
            f"n_epochs ({n_epochs}) is shorter than Stage 1 ({stage1_epochs})"
        )
    if stage2_epochs > 0 and n_epochs < stage1_epochs + stage2_epochs:
        raise ValueError(
            "n_epochs must cover both Stage 1 and Stage 2 before joint fine-tuning"
        )


def phase_for_epoch(epoch, stage1_epochs, stage2_epochs):
    """Return the DMIL stage for a one-based epoch, or ``None`` when done.

    A zero-length Stage 2 intentionally disables both Stage 2 and Stage 3,
    because the paper's joint fine-tuning stage depends on the decomposition
    learned in Stage 2.
    """
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise TypeError(f"epoch must be an integer, got {epoch!r}")
    if epoch <= 0:
        raise ValueError("epoch must be greater than zero")
    if stage1_epochs <= 0:
        raise ValueError("stage1_epochs must be greater than zero")
    if stage2_epochs < 0:
        raise ValueError("stage2_epochs must be non-negative")

    if epoch <= stage1_epochs:
        return 1
    if stage2_epochs == 0:
        return None
    if epoch <= stage1_epochs + stage2_epochs:
        return 2
    return 3
