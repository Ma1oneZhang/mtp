"""Muon + plain CE probe for the convolution-only DFlash2 ablation."""

from train_probe import main
from train_probe_muon_dflash2_conv import add_dflash2_convolution

if __name__ == "__main__":
    main(
        optimizer_name="muon",
        loss_name="ce",
        architecture_name="dflash2_conv",
        draft_transform=add_dflash2_convolution,
    )
