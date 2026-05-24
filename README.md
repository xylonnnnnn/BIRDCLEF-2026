# BirdCLEF+ 2026 — Bioacoustic Species Classification

This repository contains my solution for the **BirdCLEF+ 2026** Kaggle competition.

The task is to detect which animal species are present in short audio segments from natural soundscape recordings. The problem is formulated as a **multi-label audio classification** task, where each 5-second audio window can contain one or more species.

## Public Score

**Public score:** `0.869`

The final submission was produced using a **3-fold ensemble**.

## Solution Overview

The solution is based on a single `timm` image backbone trained on log-mel spectrograms.

Pipeline:

```text
Audio file
→ 5-second audio chunk
→ log-mel spectrogram
→ EfficientNetV2-B0 backbone
→ linear classification head
→ sigmoid probabilities
→ fold ensemble + TTA
→ submission.csv
