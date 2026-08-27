"""Muon entry point for the one-GPU DFlash training probe."""

from train_probe import main

if __name__ == "__main__":
    main(optimizer_name="muon", loss_name="pal", pal_ce_weight=0.10)
