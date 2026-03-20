#!/usr/bin/env python3
"""Create a code-aligned diagram for the multimodal monster model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_box(ax, xy, width, height, text, fc, ec="#1f2937", fontsize=11):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.8,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        wrap=True,
    )


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2,
        color="#374151",
    )
    ax.add_patch(arrow)


def main() -> None:
    output_path = Path(__file__).resolve().parent / "monster_model_diagram.png"

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        (0.03, 0.72),
        0.24,
        0.15,
        "StayFusionDataset\nstay_sequences_2048.parquet\nEHR tokens per stay_id\nconcept_ids + type_ids + time_stamps",
        "#dbeafe",
    )
    add_box(
        ax,
        (0.03, 0.43),
        0.24,
        0.15,
        "CDE inputs\ncde_coeffs.pt + cde_meta.parquet\ncontinuous ICU physiology\naligned by stay_id",
        "#dcfce7",
    )
    add_box(
        ax,
        (0.03, 0.14),
        0.24,
        0.15,
        "CXR inputs\nstay_image_index.parquet\nprecomputed tensor packs or JPG paths\n1+ studies per stay_id",
        "#fde68a",
    )

    add_box(
        ax,
        (0.34, 0.72),
        0.22,
        0.15,
        "StayEHRMambaEncoder\ninitialized from MambaPretrain\nold patient-level backbone -> stay-level EHR encoder",
        "#bfdbfe",
    )
    add_box(
        ax,
        (0.34, 0.43),
        0.22,
        0.15,
        "StayNCDEEncoder\ntrajectory encoder\npooled + sequence outputs",
        "#bbf7d0",
    )
    add_box(
        ax,
        (0.34, 0.14),
        0.22,
        0.15,
        "StayImageEncoder\nCXR branch\npooled + sequence outputs",
        "#fcd34d",
    )

    add_box(
        ax,
        (0.62, 0.38),
        0.24,
        0.23,
        "Fusion model in train_stay_fusion.py\nCrossAttentionFusionModel\nor GatedFusionModel / LateFusionModel\nmodality mask: ehr + cde + img",
        "#fbcfe8",
    )
    add_box(
        ax,
        (0.62, 0.10),
        0.24,
        0.17,
        "Multi-task logits\nlabel_in_hosp_mortality\nlabel_mortality_28d\nlabel_sepsis",
        "#e9d5ff",
    )
    add_box(
        ax,
        (0.89, 0.38),
        0.08,
        0.23,
        "Experiments\nEHR only\nEHR+CDE\nEHR+IMG\nEHR+CDE+IMG",
        "#fecaca",
        fontsize=10,
    )

    add_arrow(ax, (0.27, 0.795), (0.34, 0.795))
    add_arrow(ax, (0.27, 0.505), (0.34, 0.505))
    add_arrow(ax, (0.27, 0.215), (0.34, 0.215))

    add_arrow(ax, (0.56, 0.795), (0.62, 0.545))
    add_arrow(ax, (0.56, 0.505), (0.62, 0.495))
    add_arrow(ax, (0.56, 0.215), (0.62, 0.445))

    add_arrow(ax, (0.74, 0.38), (0.74, 0.27))
    add_arrow(ax, (0.86, 0.495), (0.89, 0.495))

    ax.text(
        0.5,
        0.95,
        "Code-Aligned Multimodal ICU \"Monster Model\"",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )

    ax.text(
        0.5,
        0.91,
        "Old Mamba backbone reused inside the stay-level fusion pipeline",
        ha="center",
        va="center",
        fontsize=12,
        color="#4b5563",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
