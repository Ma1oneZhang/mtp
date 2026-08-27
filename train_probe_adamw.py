"""AdamW entry point for the one-GPU DFlash training probe."""

from train_probe import main

if __name__ == "__main__":
    main(optimizer_name="adamw")
