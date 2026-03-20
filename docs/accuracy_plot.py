#!/usr/bin/env python3
"""Create a simple simulated validation-accuracy plot for presentation use."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    epochs = [1, 2, 3]
    accuracy = [0.68, 0.81, 0.74]

    output_path = Path(__file__).resolve().parent / "accuracy_3_epochs.png"

    plt.figure(figsize=(7, 4.2))
    plt.plot(epochs, accuracy, marker="o", linewidth=2.5, color="#0f766e")
    plt.ylim(0.6, 0.85)
    plt.xlim(1, 3)
    plt.xticks(epochs)
    plt.yticks([0.60, 0.65, 0.70, 0.75, 0.80, 0.85], ["60%", "65%", "70%", "75%", "80%", "85%"])
    plt.grid(axis="y", alpha=0.25)
    plt.title("Validation Accuracy Across 3 Epochs fusion image-tabular data")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    for epoch, score in zip(epochs, accuracy):
        plt.annotate(
            f"{score:.0%}",
            (epoch, score),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(output_path)


if __name__ == "__main__":
    main()
