"""
Image-level GD coverage analysis for the final six-channel proposed model.

Run this as a new Kaggle cell after the complete final-model cell has created:
    model, test_loader, device, MEAN, STD

This script performs no-TTA inference only. It does not retrain the model.
Outputs: per-image CSV, summary JSON, scatter plot, representative figure,
and one downloadable ZIP archive.
"""

from pathlib import Path
import csv
from datetime import datetime
import json
import math
import secrets
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1. Requirements and settings
# -----------------------------------------------------------------------------

REQUIRED_OBJECTS = ["model", "test_loader", "device", "MEAN", "STD"]
missing_objects = [name for name in REQUIRED_OBJECTS if name not in globals()]
if missing_objects:
    raise RuntimeError(
        "Run the complete final six-channel model/checkpoint-loading cell first. "
        "Missing: " + ", ".join(missing_objects)
    )

# This is the validation-selected no-TTA threshold of the final seed-42 model.
# Do not tune this value on the test set.
VALIDATION_THRESHOLD = 0.60

WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
COVERAGE_OUT_DIR = WORK_DIR / "gd_image_level_coverage_seed42"
COVERAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

mean_array = np.asarray(MEAN, dtype=np.float32).reshape(1, 1, -1)
std_array = np.asarray(STD, dtype=np.float32).reshape(1, 1, -1)
if mean_array.shape[-1] != 6 or std_array.shape[-1] != 6:
    raise ValueError("This analysis requires the final six-channel model statistics.")

model.eval()


def extract_logits(model_output):
    """Accept a tensor or a tuple/list whose first item is the final logits."""
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, (tuple, list)) and model_output:
        if torch.is_tensor(model_output[0]):
            return model_output[0]
    if isinstance(model_output, dict):
        for key in ("logits", "out", "prediction"):
            if key in model_output and torch.is_tensor(model_output[key]):
                return model_output[key]
    raise TypeError("Could not identify segmentation logits from model output.")


def normalize_names(names, batch_size):
    if isinstance(names, str):
        return [names]
    if isinstance(names, (list, tuple)):
        return [str(item) for item in names]
    return [f"sample_{index:04d}" for index in range(batch_size)]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denominator = np.sum((y_true - y_true.mean()) ** 2)
    if denominator == 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator)


def per_image_dice(prediction, target):
    intersection = np.logical_and(prediction, target).sum()
    denominator = prediction.sum() + target.sum()
    return float((2.0 * intersection + 1e-7) / (denominator + 1e-7))


# -----------------------------------------------------------------------------
# 2. No-TTA inference and image-level coverage calculation
# -----------------------------------------------------------------------------

coverage_rows = []

with torch.inference_mode():
    for batch in test_loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise RuntimeError("test_loader must return images and masks.")

        images, masks = batch[0], batch[1]
        names = batch[2] if len(batch) > 2 else None
        batch_names = normalize_names(names, images.shape[0])

        images_device = images.to(device, non_blocking=True)
        masks_device = masks.to(device, non_blocking=True)

        logits = extract_logits(model(images_device))
        if logits.shape[-2:] != masks_device.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks_device.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        probabilities = torch.sigmoid(logits)
        predictions = probabilities >= VALIDATION_THRESHOLD
        targets = masks_device >= 0.5

        for index in range(images.shape[0]):
            probability = probabilities[index, 0].detach().cpu().numpy()
            prediction = predictions[index, 0].detach().cpu().numpy().astype(bool)
            target = targets[index, 0].detach().cpu().numpy().astype(bool)

            gt_coverage = float(target.mean() * 100.0)
            predicted_coverage = float(prediction.mean() * 100.0)
            signed_error = predicted_coverage - gt_coverage

            coverage_rows.append(
                {
                    "image": batch_names[index],
                    "ground_truth_coverage_percent": gt_coverage,
                    "predicted_coverage_percent": predicted_coverage,
                    "signed_error_percentage_points": signed_error,
                    "absolute_error_percentage_points": abs(signed_error),
                    "squared_error": signed_error ** 2,
                    "dice_at_validation_threshold": per_image_dice(prediction, target),
                }
            )

if not coverage_rows:
    raise RuntimeError("No test samples were processed.")

gt_values = np.asarray(
    [row["ground_truth_coverage_percent"] for row in coverage_rows],
    dtype=np.float64,
)
predicted_values = np.asarray(
    [row["predicted_coverage_percent"] for row in coverage_rows],
    dtype=np.float64,
)
errors = predicted_values - gt_values

coverage_summary = {
    "evaluation": "test set, no TTA",
    "number_of_test_images": int(len(coverage_rows)),
    "validation_selected_threshold": float(VALIDATION_THRESHOLD),
    "coverage_denominator": "all pixels in the full image",
    "mae_percentage_points": float(np.mean(np.abs(errors))),
    "rmse_percentage_points": float(np.sqrt(np.mean(errors ** 2))),
    "mean_signed_error_percentage_points": float(np.mean(errors)),
    "pearson_r": safe_pearson(gt_values, predicted_values),
    "r_squared": safe_r2(gt_values, predicted_values),
    "mean_ground_truth_coverage_percent": float(np.mean(gt_values)),
    "mean_predicted_coverage_percent": float(np.mean(predicted_values)),
}

write_csv(COVERAGE_OUT_DIR / "per_image_gd_coverage.csv", coverage_rows)
with open(COVERAGE_OUT_DIR / "coverage_summary.json", "w", encoding="utf-8") as file:
    json.dump(coverage_summary, file, indent=2, allow_nan=True)


# -----------------------------------------------------------------------------
# 3. Paper-ready scatter plot
# -----------------------------------------------------------------------------

minimum_axis = min(0.0, float(gt_values.min()), float(predicted_values.min()))
maximum_axis = max(float(gt_values.max()), float(predicted_values.max()))
axis_padding = max(0.25, 0.06 * max(maximum_axis - minimum_axis, 1.0))
axis_low = minimum_axis - axis_padding
axis_high = maximum_axis + axis_padding

figure, axis = plt.subplots(figsize=(6.2, 5.6))
axis.scatter(
    gt_values,
    predicted_values,
    s=46,
    color="#2C7FB8",
    edgecolor="white",
    linewidth=0.7,
    alpha=0.88,
)
axis.plot(
    [axis_low, axis_high],
    [axis_low, axis_high],
    linestyle="--",
    linewidth=1.4,
    color="#444444",
    label="Ideal agreement",
)
axis.set_xlim(axis_low, axis_high)
axis.set_ylim(axis_low, axis_high)
axis.set_aspect("equal", adjustable="box")
axis.set_xlabel("Ground-truth image-level GD coverage (%)")
axis.set_ylabel("Predicted image-level GD coverage (%)")
axis.grid(alpha=0.22, linewidth=0.7)
axis.legend(loc="lower right", frameon=False)
axis.text(
    0.04,
    0.96,
    "MAE = {mae:.3f} pp\nRMSE = {rmse:.3f} pp\n"
    "Pearson r = {r:.3f}\n$R^2$ = {r2:.3f}".format(
        mae=coverage_summary["mae_percentage_points"],
        rmse=coverage_summary["rmse_percentage_points"],
        r=coverage_summary["pearson_r"],
        r2=coverage_summary["r_squared"],
    ),
    transform=axis.transAxes,
    va="top",
    ha="left",
    fontsize=10,
    bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
)
figure.tight_layout()
scatter_png = COVERAGE_OUT_DIR / "gd_coverage_scatter.png"
scatter_pdf = COVERAGE_OUT_DIR / "gd_coverage_scatter.pdf"
figure.savefig(scatter_png, dpi=400, bbox_inches="tight")
figure.savefig(scatter_pdf, bbox_inches="tight")
plt.show()
plt.close(figure)


# -----------------------------------------------------------------------------
# 4. Randomly select one image from each coverage group
# -----------------------------------------------------------------------------

positive_rows = [
    row for row in coverage_rows if row["ground_truth_coverage_percent"] > 0
]
positive_rows.sort(key=lambda row: row["ground_truth_coverage_percent"])
if not positive_rows:
    positive_rows = sorted(
        coverage_rows,
        key=lambda row: row["ground_truth_coverage_percent"],
    )

# Keep the lower/intermediate/higher structure, but choose a random image from
# each third of the positive-coverage test set. The last selection is excluded
# so that consecutive runs do not display the same examples when alternatives
# are available. This randomness affects only the figure, not any metric.
coverage_groups = [
    list(group)
    for group in np.array_split(np.asarray(positive_rows, dtype=object), 3)
    if len(group) > 0
]

selection_history_path = COVERAGE_OUT_DIR / "last_random_selection.json"
last_selected_names = set()
if selection_history_path.exists():
    try:
        with open(selection_history_path, "r", encoding="utf-8") as file:
            previous_selection = json.load(file)
        last_selected_names = set(previous_selection.get("selected_images", []))
    except (OSError, ValueError, TypeError):
        last_selected_names = set()

selected_names = []
for group in coverage_groups:
    group_rows = list(group)
    unused_rows = [
        row for row in group_rows if row["image"] not in last_selected_names
    ]
    candidate_rows = unused_rows if unused_rows else group_rows
    selected_names.append(secrets.choice(candidate_rows)["image"])

with open(selection_history_path, "w", encoding="utf-8") as file:
    json.dump(
        {
            "run_id": RUN_ID,
            "selected_images": selected_names,
            "selection_rule": (
                "one random positive-GD image from each coverage third"
            ),
        },
        file,
        indent=2,
    )

print("Random examples selected:", ", ".join(selected_names))

selected_name_set = set(selected_names)
visual_records = []

with torch.inference_mode():
    for batch in test_loader:
        images, masks = batch[0], batch[1]
        names = batch[2] if len(batch) > 2 else None
        batch_names = normalize_names(names, images.shape[0])

        wanted_indices = [
            index for index, name in enumerate(batch_names)
            if name in selected_name_set
        ]
        if not wanted_indices:
            continue

        images_device = images.to(device, non_blocking=True)
        logits = extract_logits(model(images_device))
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        probabilities = torch.sigmoid(logits).detach().cpu()

        for index in wanted_indices:
            normalized_image = images[index].permute(1, 2, 0).cpu().numpy()
            original_image = normalized_image * std_array + mean_array
            rgb = np.clip(original_image[..., :3], 0.0, 1.0)
            probability = probabilities[index, 0].numpy()
            prediction = probability >= VALIDATION_THRESHOLD
            target = masks[index, 0].cpu().numpy() >= 0.5
            row = next(item for item in coverage_rows if item["image"] == batch_names[index])

            visual_records.append(
                {
                    "image": batch_names[index],
                    "rgb": rgb,
                    "target": target,
                    "prediction": prediction,
                    "probability": probability,
                    "gt_coverage": row["ground_truth_coverage_percent"],
                    "predicted_coverage": row["predicted_coverage_percent"],
                    "dice": row["dice_at_validation_threshold"],
                }
            )

visual_records.sort(key=lambda row: row["gt_coverage"])

if visual_records:
    level_names = ["Lower coverage", "Intermediate coverage", "Higher coverage"]
    figure, axes = plt.subplots(
        len(visual_records),
        4,
        figsize=(12.8, 3.2 * len(visual_records)),
        squeeze=False,
    )
    probability_artist = None
    column_titles = ["RGB", "Ground truth", "Prediction", "GD probability"]

    for row_index, record in enumerate(visual_records):
        axes[row_index, 0].imshow(record["rgb"])
        axes[row_index, 1].imshow(record["target"], cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].imshow(record["prediction"], cmap="gray", vmin=0, vmax=1)
        probability_artist = axes[row_index, 3].imshow(
            record["probability"], cmap="magma", vmin=0.0, vmax=1.0
        )

        row_label = level_names[min(row_index, len(level_names) - 1)]
        axes[row_index, 0].set_ylabel(
            f"{row_label}\nGT={record['gt_coverage']:.2f}% | "
            f"Pred={record['predicted_coverage']:.2f}%\n"
            f"Dice={record['dice']:.3f}",
            fontsize=9,
        )

        for column_index, title in enumerate(column_titles):
            if row_index == 0:
                axes[row_index, column_index].set_title(title, fontsize=10)
            axes[row_index, column_index].set_xticks([])
            axes[row_index, column_index].set_yticks([])

    if probability_artist is not None:
        colorbar = figure.colorbar(
            probability_artist,
            ax=axes[:, 3].ravel().tolist(),
            fraction=0.028,
            pad=0.02,
        )
        colorbar.set_label("Predicted GD probability")

    figure.subplots_adjust(
        left=0.13, right=0.92, top=0.94, bottom=0.04,
        wspace=0.04, hspace=0.13,
    )
    examples_png = COVERAGE_OUT_DIR / (
        f"random_gd_coverage_examples_{RUN_ID}.png"
    )
    examples_pdf = COVERAGE_OUT_DIR / (
        f"random_gd_coverage_examples_{RUN_ID}.pdf"
    )
    figure.savefig(examples_png, dpi=400, bbox_inches="tight")
    figure.savefig(examples_pdf, bbox_inches="tight")
    plt.show()
    plt.close(figure)


# -----------------------------------------------------------------------------
# 5. Print results and create one downloadable archive
# -----------------------------------------------------------------------------

print("\nIMAGE-LEVEL GD COVERAGE RESULTS")
print("Evaluation             : test set, no TTA")
print(f"Validation threshold   : {VALIDATION_THRESHOLD:.2f}")
print(f"Test images            : {len(coverage_rows)}")
print(f"MAE                    : {coverage_summary['mae_percentage_points']:.4f} pp")
print(f"RMSE                   : {coverage_summary['rmse_percentage_points']:.4f} pp")
print(f"Mean signed error      : {coverage_summary['mean_signed_error_percentage_points']:.4f} pp")
print(f"Pearson r              : {coverage_summary['pearson_r']:.4f}")
print(f"R-squared              : {coverage_summary['r_squared']:.4f}")
print("Coverage denominator   : all pixels in the full image")
print("Output directory       :", COVERAGE_OUT_DIR)

archive_base = WORK_DIR / "gd_image_level_coverage_seed42"
archive_path = Path(
    shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=COVERAGE_OUT_DIR,
    )
)
print("ZIP                    :", archive_path)

try:
    from IPython.display import FileLink, display
    display(FileLink(str(archive_path)))
except Exception:
    pass