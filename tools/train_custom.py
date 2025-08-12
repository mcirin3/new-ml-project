# tools/train_custom.py
from src.custom_model import train_custom

if __name__ == "__main__":
    path, summary = train_custom()
    print("Saved model to:", path)
    for k, v in summary.items():
        print(f"{k}: {v}")
