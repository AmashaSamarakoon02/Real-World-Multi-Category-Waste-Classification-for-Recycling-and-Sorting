# Real-World Multi-Category Waste Classification — Code Base

This code implements the methodology described in your proposal
*"Real-World Multi-Category Waste Classification for Recycling and Sorting"*
(Figure 1 / Section 3.2). Every script maps to one objective from Section 1.3.

## 1. Setup

```bash
pip install -r requirements.txt
```

Unzip your two datasets so the folder layout matches `config.py`:

```
data/
  trashnet/                 <- from dataset-resized.zip (already matches: cardboard, glass, metal, paper, plastic, trash)
    cardboard/*.jpg
    glass/*.jpg
    metal/*.jpg
    paper/*.jpg
    plastic/*.jpg
    trash/*.jpg
  realwaste/                <- from the RealWaste GitHub repo (the folder you screenshotted)
    Cardboard/*.jpg
    Food Organics/*.jpg
    Glass/*.jpg
    Metal/*.jpg
    Miscellaneous Trash/*.jpg
    Paper/*.jpg
    Plastic/*.jpg
    Textile Trash/*.jpg
    Vegetation/*.jpg
```

**This zip already contains both datasets, pre-loaded and folder-matched:**
- `data/trashnet/` — the full TrashNet dataset (2,527 images across all 6 classes),
  unzipped straight from your `dataset-resized.zip` (its folder names already matched
  `config.TRASHNET_CLASSES` exactly, no renaming needed).
- `data/realwaste/` — a **small sample** of RealWaste (4-7 images per class, 40 images
  total) from your `Real.zip` upload, with folder names normalised to match
  `config.REALWASTE_CLASSES` exactly (e.g. `cardboard` → `Cardboard`,
  `Food organics` → `Food Organics`, `paper` → `Paper`, `plastic` → `Plastic`).

⚠️ **The RealWaste sample is a placeholder, not the full dataset.** 40 images is
enough to confirm the whole pipeline runs end-to-end (loading, splitting, zero-shot
evaluation, fine-tuning) without errors, but it is far too small to produce a
statistically meaningful clean-to-real accuracy gap or reliable confidence intervals
(Section 3.2.4). Before running real experiments for your dissertation, replace
`data/realwaste/` with the full dataset from
[Kaggle: RealWaste](https://www.kaggle.com/datasets/joebeachcapital/realwaste)
(~4,752 images) — same folder-per-class layout, so it will drop in with zero code
changes. You can also set `TRASHNET_DIR` / `REALWASTE_DIR` environment variables
instead of physically replacing the folders.

## 2. Pipeline — run in this order

Each step is a Jupyter notebook — open it and choose **Run All**.

| Step | Notebook | Proposal reference |
|---|---|---|
| 1 | `train_trashnet.ipynb` | **Objective 1** — trains ResNet50, EfficientNet-B0, MobileNetV3-Large on the clean TrashNet baseline (70/15/15 split, Table 2). Run this once per backbone (`train_one_backbone("resnet50")`, etc.), or run the final cell to train all three in sequence. |
| 2 | `evaluate_cross_domain.ipynb` | **Objective 2 / RQ1** — zero-shot test of a TrashNet-trained model on RealWaste (no retraining), reports the clean-to-real accuracy gap. Set `BACKBONE` in the second cell, or run the final cell to evaluate all three. |
| 3 | `domain_generalization.ipynb` | **Objective 3 / RQ2** — heavy augmentation + mixup fine-tuning on a small reserved RealWaste subset (20% by default, disjoint from the zero-shot test set) to close the gap. Set `TECHNIQUE` to `"heavy_aug"`, `"mixup"`, or `"both"` to isolate each technique. |
| 4 | `compare_backbones.ipynb` | **Objective 4 / RQ3** — builds the accuracy-vs-speed comparison table + chart across all three backbones (reads the JSON results saved by steps 1–2). |
| 5 | `confidence_analysis.ipynb` | **Section 3.2.4** — bootstrap-resampled per-class confidence intervals, following the RealWaste reliability-analysis literature. |
| 6 | `scorecam.ipynb` | **Section 3.1 / Objective 5 evaluation** — Score-CAM heatmap confirming the model looks at the waste item, not the background. Set `IMAGE_PATH` to any RealWaste/TrashNet image. |
| 7 | `classify_tool.ipynb` | **Objective 5 / RQ4** — the locally runnable classification tool (CLI + Tkinter GUI), for both household and sorting-line use. Run `run_cli(...)` headlessly or `run_gui(...)` for the desktop window. |

Steps 1 must complete before step 2, 3, 6, or 7 (they all load a trained TrashNet checkpoint). Step 4 needs the JSON results from steps 1–2 to be non-empty.

## 3. File map

```
config.py                    Paths, class lists, TrashNet<->RealWaste category mapping, hyperparameters
dataset.py                   Dataset loading, clean/heavy transforms, splits, mixup
models.py                    ResNet50 / EfficientNet-B0 / MobileNetV3-Large builders
scorecam.py                  Score-CAM explainability (shared module — imported by scorecam.ipynb and classify_tool.ipynb)
train_trashnet.ipynb         Objective 1
evaluate_cross_domain.ipynb  Objective 2 (zero-shot clean-to-real gap)
domain_generalization.ipynb  Objective 3 (heavy aug / mixup / fine-tuning)
compare_backbones.ipynb      Objective 4 (accuracy vs. inference speed)
confidence_analysis.ipynb    Bootstrap confidence-score reliability analysis
classify_tool.ipynb          Objective 5 (locally runnable CLI + Tkinter GUI tool)
checkpoints/                 Saved model weights (created automatically)
outputs/                     JSON results, CSVs, charts (created automatically)
```

## 4. Notes on the category mismatch

TrashNet has 6 classes; RealWaste has 9. Three RealWaste categories
(`Food Organics`, `Textile Trash`, `Vegetation`) do not exist in TrashNet's label
space, so they are **excluded** from the strict zero-shot cross-domain accuracy
number (Objective 2) — including them would make the "gap" partly an artefact of
missing classes rather than genuine domain shift. They **are** included in the
Objective 3 fine-tuning stage, where the classifier head is expanded to all 9
RealWaste classes and adapted directly. This mirrors the "clean-to-real gap must
reflect domain shift, not category definitions" principle from Section 3.1 of
your proposal.

## 5. Maintenance notes

This code base was reviewed and repaired to fix a set of `.py` → `.ipynb`
conversion artifacts (broken indentation, a broken cross-notebook import, and
one silent dictionary-key mismatch) that previously prevented several
notebooks from running. No methodology, folder structure, or research logic
was changed — see the accompanying fix summary for the full list. In
particular, `scorecam.py` was restored as a proper shared module (Score-CAM
was previously only defined inline inside `scorecam.ipynb`, which meant
`classify_tool.ipynb`'s `from scorecam import ScoreCAM, overlay_heatmap`
could never succeed).

## 6. What still needs a GPU / real run

This environment has no GPU and no network access, so nothing here has actually
been trained — the code is complete and internally consistent (uses your uploaded
TrashNet folder structure directly), but you'll need to run it on your own machine
or Colab/Kaggle with a GPU. `config.DEVICE` auto-detects CUDA and falls back to CPU.
