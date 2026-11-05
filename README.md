<div align="center">

# ResNet-50 versus ViT-Base for Tuberculosis Bacilli Detection in Ziehl-Neelsen-Stained Sputum Smears

**A controlled, leakage-aware comparison of a convolutional and a transformer architecture on public smear microscopy images**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-1.0.29-1F6FEB?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Google Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

[![GitHub repository](https://img.shields.io/badge/Code-GitHub%20repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/st10251759/RPDA8412---Research-Project---notebook)
[![Google Drive folder](https://img.shields.io/badge/Notebook-Google%20Drive%20folder-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1_cEhWtv40TbBO1G7IbB0Fi_NK6wHljKa?usp=sharing)

![Seed](https://img.shields.io/badge/Random%20seed-42-4B5563?style=flat-square)
![Test set](https://img.shields.io/badge/Test%20set-evaluated%20once-2E7D32?style=flat-square)
![Design](https://img.shields.io/badge/Design-group--aware%205--fold%20CV-0B7285?style=flat-square)
![Module](https://img.shields.io/badge/Module-RPDA8412-6D28D9?style=flat-square)
![Level](https://img.shields.io/badge/NQF%20Level%208-Data%20Analytics-1D4ED8?style=flat-square)

</div>

---

## Contents

1. [Project summary](#1-project-summary)
2. [Research questions and hypotheses](#2-research-questions-and-hypotheses)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
6. [Interpretation and limits](#6-interpretation-and-limits)
7. [Repository contents](#7-repository-contents)
8. [Setup and how to run the notebook](#8-setup-and-how-to-run-the-notebook)
9. [Reproducibility record](#9-reproducibility-record)
10. [Code attributions](#10-code-attributions)
11. [Data, ethics and responsible use](#11-data-ethics-and-responsible-use)
12. [Project links and how to cite](#12-project-links-and-how-to-cite)

---

## 1. Project summary

Ziehl-Neelsen (ZN) sputum smear microscopy is still used to diagnose and monitor pulmonary tuberculosis where molecular testing is limited. Reading a smear by eye is slow and depends on the reader. This repository holds the analysis notebook for a research project that asks which of two widely used image classifiers is better suited to the first automated step in that process: deciding whether a microscope field contains acid-fast bacilli.

The two architectures were compared under identical conditions:

| | ResNet-50 | ViT-Base |
|---|---|---|
| Family | Convolutional neural network with residual connections | Vision Transformer using self-attention over 16 by 16 pixel patches |
| Weights | torchvision `IMAGENET1K_V2` | timm `vit_base_patch16_224.augreg_in1k` |
| Input size | 224 by 224 pixels | 224 by 224 pixels |
| Task | Binary classification of each image: bacilli present or absent | Binary classification of each image: bacilli present or absent |

The study is a research comparison on one public dataset. It is not a clinical validation, and its outputs must not be used for patient diagnosis.

**Headline finding.** ViT-Base had higher point estimates on every reported test metric, but none of the four pre-specified comparisons was statistically significant after Bonferroni adjustment, and the test set held only six negative images. The evidence therefore does not show that either architecture is superior, and it does not show that they are equivalent.

---

## 2. Research questions and hypotheses

The study compared the two models on four held-out test metrics. Each hypothesis was tested at a Bonferroni-adjusted significance level of 0.0125 (0.05 divided by four primary tests).

| Hypothesis | Metric | Test used | Null hypothesis |
|---|---|---|---|
| H1 | Accuracy | McNemar test | The two models have equal accuracy |
| H2 | Sensitivity | McNemar test | The two models have equal sensitivity |
| H3 | Specificity | Exact McNemar test | The two models have equal specificity |
| H4 | AUC-ROC | DeLong test | The two models have equal AUC-ROC |

A fifth question (SQ5) examined, descriptively, where each model attended in the image, using Grad-CAM for ResNet-50 and attention maps for ViT-Base. This part is exploratory and is not evidence of clinical validity.

---

## 3. Dataset

| Item | Value |
|---|---|
| Source | Public Kaggle dataset: Saife (2020), `saife245/tuberculosis-image-datasets` |
| Files verified | 1,265 images and 1,265 matching XML annotation files |
| Annotated bacilli | 9,969 bounding boxes |
| Images analysed | 1,247 (1,218 positive, 29 negative) |
| Class ratio | About 42 positive images for every negative image |
| Training and validation partition | 994 images |
| Held-out test partition | 253 images (247 positive, 6 negative) |

The dataset page lists 928 images and 3,734 bacilli. File-level verification found the larger counts above, and every figure in this repository uses the verified counts. Every excluded image is recorded with a reason in `exclusions.csv`. The images are not redistributed here; the notebook downloads them from the original source.

The negative class is small. This is the main constraint on the specificity results and is discussed in [Section 6](#6-interpretation-and-limits).

---

## 4. Methodology

The pipeline has seven stages. Each stage writes its outputs to disk, so any number reported in the project can be traced to a saved file.

### 4.1 Data verification

Every image and XML file was parsed. The checks covered missing or orphan annotation files, unreadable images, byte-identical duplicates, and the per-image bacillus counts and box geometry. Images with no annotation file were excluded. A first-pass image-quality review of zero-annotation images was carried out on the original microscopy images, and the outcome is logged in `image_review.csv`.

### 4.2 Leakage control

Microscope fields from the same slide can look almost identical, so a random split could place near copies in both training and test data. To prevent this, each image was reduced to an 8 by 8 average hash, and images within a Hamming distance of 5 were linked into clusters. Whole clusters were then assigned to a single partition, so no near-duplicate cluster crosses the training, validation and test boundary. The pairs are listed in `near_duplicate_pairs.csv` and the assignment in `split_assignment.csv`.

### 4.3 Split and cross-validation

- An 80:20 group-aware, class-stratified split produced the 994-image training and validation partition and the 253-image test partition.
- Five-fold group-aware cross-validation was run inside the 994-image partition only.
- The test partition was locked until model selection was finished. The file `TEST_SET_UNLOCKED.txt` records the moment it was opened.

### 4.4 Preprocessing and augmentation

- Images were resized to 224 by 224 pixels and converted to tensors.
- Normalisation was matched to each backbone: ImageNet channel means and standard deviations for ResNet-50, and a mean and standard deviation of 0.5 for ViT-Base. This is the only preprocessing step that differs between the two models.
- Augmentation was applied to training images only: horizontal and vertical flips (probability 0.5), rotation up to 15 degrees, random resized crop and colour jitter.
- Two augmentation intensities were part of the search grid:

| Intensity | Crop scale | Colour jitter |
|---|---|---|
| Low | 0.9 to 1.0 | 0.1 |
| Moderate | 0.8 to 1.0 | 0.2 |

### 4.5 Two-stage transfer learning

| Setting | Stage 1 | Stage 2 |
|---|---|---|
| What is trained | Classification head only, backbone frozen | Whole network |
| Learning rate | 0.001 | Searched: 0.0001, 0.00005, 0.00001 |
| Epoch budget | Up to 10 | Remaining budget, 50 epochs in total |
| Early stopping patience | 3 epochs | 7 epochs |
| Scheduler | None | Cosine annealing over the remaining epochs |

Shared settings: Adam optimiser, batch size 32, mixed-precision training, and a binary cross-entropy loss with logits. The loss weights the positive class by the ratio of negative to positive images in the training data, which counters the class imbalance. The decision threshold was fixed at 0.5 and was not tuned on the test set.

### 4.6 Hyperparameter search

A small grid was searched for each model with five-fold cross-validation:

| Hyperparameter | Values |
|---|---|
| Stage 2 learning rate | 0.0001, 0.00005, 0.00001 |
| Weight decay | 0.0001, 0.00001 |
| Augmentation intensity | Low, moderate |

This gives 12 configurations per model, 60 training runs per model and 120 in total. The configuration with the highest mean validation AUC across folds was selected, with a lower standard deviation across folds as the tie-break. The selected configurations are stored in `resnet50_selected_config.json` and `vit_base_selected_config.json`.

| | ResNet-50 | ViT-Base |
|---|---|---|
| Selected configuration | LR 0.0001, weight decay 0.00001, moderate | LR 0.00005, weight decay 0.0001, moderate |
| Mean validation AUC (5 folds) | 0.8326 | 0.9195 |
| Standard deviation across folds | 0.1237 | 0.0931 |
| Final retraining epochs (Stage 1, Stage 2) | 3, 0 | 5, 6 |

Each selected configuration was retrained once on all 994 training and validation images, using the median best epoch across folds for each stage. The median Stage 2 epoch for ResNet-50 was zero, so the final ResNet-50 model used only the frozen-backbone stage. This follows from the stopping rule and is reported openly as a limitation of the ResNet-50 result.

### 4.7 Held-out evaluation and statistics

Both final models were scored once on the 253 test images.

- **Point metrics:** accuracy, sensitivity, specificity, precision, F1 and AUC-ROC.
- **Confidence intervals:** Wilson intervals for proportions, and 2,000-resample stratified bootstrap intervals for AUC-ROC and F1.
- **Paired tests:** McNemar for accuracy, sensitivity and specificity (exact binomial form when the discordant count is below 25) and DeLong for AUC-ROC.
- **Multiplicity:** Bonferroni adjustment across the four primary tests (alpha 0.0125).
- **Sensitivity analyses:** a group-level bootstrap that resamples whole near-duplicate clusters, so the intervals do not assume that test images are independent.

### 4.8 Interpretability subset

Seventeen test images (12 positive, 5 negative) were selected for a descriptive look at model attention: Grad-CAM for ResNet-50 and attention maps for ViT-Base. Attention maps show where a model looked, not why it decided, and these outputs are exploratory.

---

## 5. Results

All numbers come from a single evaluation of each final model on the 253-image test partition. Intervals are 95% intervals.

### 5.1 Test performance

| Metric | ResNet-50 | 95% CI | ViT-Base | 95% CI |
|---|---|---|---|---|
| Accuracy | 0.8300 | 0.7789 to 0.8713 | 0.8617 | 0.8137 to 0.8988 |
| Sensitivity | 0.8340 | 0.7826 to 0.8752 | 0.8623 | 0.8138 to 0.8998 |
| Specificity | 0.6667 | 0.3000 to 0.9032 | 0.8333 | 0.4365 to 0.9699 |
| Precision | 0.9904 | 0.9656 to 0.9974 | 0.9953 | 0.9740 to 0.9992 |
| F1 | 0.9055 | 0.8764 to 0.9313 | 0.9241 | 0.8978 to 0.9489 |
| AUC-ROC | 0.8671 | 0.7753 to 0.9501 | 0.9541 | 0.8988 to 0.9960 |

### 5.2 Hypothesis tests

| Hypothesis | Images | Difference (ResNet-50 minus ViT-Base) | 95% CI of difference | Test statistic | p-value | Decision at alpha 0.0125 |
|---|---|---|---|---|---|---|
| H1 accuracy | 253 | -0.0316 | -0.0830 to 0.0198 | 1.1136 | 0.2913 | Fail to reject |
| H2 sensitivity | 247 | -0.0283 | -0.0810 to 0.0243 | 0.8372 | 0.3602 | Fail to reject |
| H3 specificity | 6 | -0.1667 | -0.5000 to 0.0000 | exact test | 1.0000 | Fail to reject |
| H4 AUC-ROC | 253 | -0.0870 | -0.1834 to 0.0093 | -1.7704 | 0.0767 | Fail to reject |

For H1, ViT-Base alone was correct on 26 images and ResNet-50 alone on 18. For H2, the corresponding counts were 25 and 18.

### 5.3 Reading the results

- ViT-Base scored higher on every point estimate, and its AUC-ROC difference of 0.087 came closest to significance (p = 0.0767).
- No comparison passed the adjusted threshold, so the data do not support a claim that either architecture is better.
- Specificity rests on six negative images, so its intervals are very wide and the specificity comparison could not have detected a real difference.
- At the default threshold, sensitivity was 0.834 for ResNet-50 and 0.862 for ViT-Base. Each model missed between 14% and 17% of positive images.
- Precision is very high for both models because 97.7% of images are positive. A rule that labelled every field positive would already be correct on 97.7% of images, so accuracy and precision should be read alongside AUC-ROC and specificity.

---

## 6. Interpretation and limits

- **Training regimes differ in effect.** The ResNet-50 final model used no fine-tuning stage, while ViT-Base used both stages. The comparison therefore reflects each architecture under its own selected protocol, not equal training length.
- **Very few negatives.** Six negative test images cannot support a specificity conclusion in either direction.
- **One source.** All images come from one public dataset. No South African images were evaluated, so the results say nothing about performance in South African laboratories.
- **Field-level labels.** Each image is labelled as containing bacilli or not. The study did not count, localise or grade bacilli.
- **Annotation quality is undocumented.** The dataset does not describe annotator qualifications or inter-annotator agreement.
- **Single seed.** A fixed seed makes the run repeatable but does not measure stability across seeds. Some GPU operations are not bit-for-bit deterministic, so a different GPU or software version can change a result.
- **Decision support only.** The findings do not support deploying either model for diagnosis.

---

## 7. Repository contents

The notebook writes the following files. Names match those used in the research report.

| Group | Files | Purpose |
|---|---|---|
| Configuration | `config.json`, `environment.txt` | Every setting used, and the frozen package versions of the run |
| Data checks | `labelled_manifest.csv`, `exclusions.csv`, `image_review.csv`, `positive_quality_audit.csv`, `gate_manifest.csv` | Verified file inventory, exclusions with reasons and review outcomes |
| Leakage control | `near_duplicate_pairs.csv`, `split_assignment.csv` | Near-duplicate pairs and the partition of every image |
| Exploratory analysis | `image_level_features.csv`, `bacilli_box_geometry.csv`, `table_5_10_eda_summary.csv` | Class balance, box geometry, colour and quality statistics |
| Model selection | `table_resnet50_cv_selected.csv`, `table_vit_base_cv_selected.csv`, `resnet50_selected_config.json`, `vit_base_selected_config.json` | Cross-validation results and chosen configurations |
| Final models | `final_resnet50.csv`, `final_vit_base.csv`, `TEST_SET_UNLOCKED.txt` | Final training logs and the record of the single test evaluation |
| Test results | `test_predictions.csv`, `test_metrics_with_ci.csv`, `test_metric_differences.csv`, `hypothesis_tests.csv` | Image-level predictions, metrics, differences and hypothesis tests |
| Interpretability | `interpretability_subset.csv` and the `fig_*_gradcam_*` and `fig_*_attention_*` figures | Exploratory attention outputs |
| Figures | `fig_*.png` | Class distribution, geometry, colour statistics, learning curves, confusion matrices, ROC curves |
| Run record | `smoke_test_timing.json`, `outputs_latest.txt` | Timing check and the append-only record of notebook output |

---

## 8. Setup and how to run the notebook

### 8.1 Requirements

| Requirement | Detail |
|---|---|
| Platform | Google Colab with a GPU runtime (the study was run on Colab) |
| Python | 3.13 on Colab for the reported run |
| Core libraries | PyTorch 2.11.0, torchvision 0.26.0, timm 1.0.29, scikit-learn 1.6.1, SciPy 1.16.3, statsmodels 0.15.0, NumPy 2.1.3, pandas 2.2.3, Matplotlib 3.10.0, Pillow 11.3.0, OpenCV 5.0.0, kagglehub 1.0.2 |
| Accounts | A Google account for Colab and Google Drive; Kaggle access for the dataset download |
| Storage | Space in Google Drive for the dataset, checkpoints, tables and figures |

The full list of package versions from the run is in `environment.txt`. It is a complete Colab environment listing, so only the libraries above are needed to reproduce the analysis.

### 8.2 Steps

1. **Clone or download** this repository from GitHub: `https://github.com/st10251759/RPDA8412---Research-Project---notebook`.
2. **Open the notebook** in Google Colab (File, then Open notebook, then the GitHub or Upload tab).
3. **Select a GPU runtime** (Runtime, then Change runtime type, then a GPU option).
4. **Mount Google Drive** when the notebook asks, and set the output folder to a directory in your own Drive. The study used a Drive folder for outputs; its link is given in the report annexure.
5. **Provide Kaggle credentials** if `kagglehub` prompts for them. The dataset is downloaded from `saife245/tuberculosis-image-datasets`.
6. **Keep `config.json` unchanged** if you want to reproduce the reported run. All seeds, grids, thresholds and statistical settings are read from this file.
7. **Run the cells from top to bottom.** Do not skip the data verification and split stages, because later stages read the files they write.
8. **Compare your outputs** with the files listed in [Section 7](#7-repository-contents) and with the results in [Section 5](#5-results).

### 8.3 Expected run behaviour

- The hyperparameter search is the longest stage because it trains 120 models, so a full run is long and may need to be resumed if a Colab session ends. Outputs are saved to disk as each stage completes.
- Results can differ in the last decimal places on a different GPU or software version. A different selected configuration is possible for the same reason.
- The test partition is meant to be scored once. Re-running the evaluation after changing a model against the test set would invalidate the comparison.

---

## 9. Reproducibility record

| Control | Setting |
|---|---|
| Global seed | 42 for Python, NumPy and PyTorch (including CUDA) and the data loaders |
| Fold seeds | 42 plus the fold number |
| Final retraining and bootstrap seed | 42 |
| Determinism | Deterministic cuDNN kernels enabled |
| Configuration | One `config.json` read by every stage |
| Logging | Append-only, timestamped log of every notebook cell |
| Split | Saved to `split_assignment.csv` and reused by every stage |
| Environment | Package versions frozen in `environment.txt` |
| Test discipline | Single evaluation, recorded in `TEST_SET_UNLOCKED.txt` |
| Bootstrap | 2,000 resamples, 95% level |

---

## 10. Code attributions

The notebook uses published documentation, library references and methodological sources. Each entry states what the source was used for.

* PyTorch ResNet-50 Model and Pretrained Weights:
   * Author: PyTorch Contributors, Linux Foundation
   * Used for: Loading the ResNet-50 architecture with the `IMAGENET1K_V2` weights, and replacing the final layer with a single-output classification head
   * Link: [torchvision.models.resnet50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
   * Date Accessed: 5 November 2026
* Vision Transformer Model Card (timm):
   * Author: Ross Wightman, Hugging Face
   * Used for: Loading `vit_base_patch16_224.augreg_in1k` and reading its pretrained input normalisation
   * Link: [vit_base_patch16_224.augreg_in1k](https://huggingface.co/timm/vit_base_patch16_224.augreg_in1k)
   * Date Accessed: 5 November 2026
* timm Library Documentation:
   * Author: Ross Wightman, Hugging Face
   * Used for: Creating the ViT-Base model and its pretrained data configuration
   * Link: [timm documentation](https://huggingface.co/docs/timm/index)
   * Date Accessed: 5 November 2026
* An Image is Worth 16x16 Words:
   * Author: Dosovitskiy et al., ICLR 2021
   * Used for: The Vision Transformer architecture compared in the study
   * Link: [An image is worth 16x16 words: transformers for image recognition at scale](https://openreview.net/forum?id=YicbFdNTTy)
   * Date Accessed: 5 November 2026
* How to Train Your ViT:
   * Author: Steiner et al., Transactions on Machine Learning Research
   * Used for: The augmentation and regularisation (AugReg) pretraining behind the ViT-Base weights
   * Link: [How to train your ViT? Data, augmentation, and regularization in vision transformers](https://arxiv.org/abs/2106.10270)
   * Date Accessed: 5 November 2026
* Loss Function with Class Weighting:
   * Author: PyTorch Contributors, Linux Foundation
   * Used for: Binary cross-entropy with logits and the `pos_weight` argument that counters class imbalance
   * Link: [torch.nn.BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
   * Date Accessed: 5 November 2026
* Adam Optimiser:
   * Author: PyTorch Contributors, Linux Foundation
   * Used for: Optimising the classification head in Stage 1 and the full network in Stage 2
   * Link: [torch.optim.Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
   * Date Accessed: 5 November 2026
* Cosine Annealing Learning Rate Schedule:
   * Author: PyTorch Contributors, Linux Foundation
   * Used for: The Stage 2 learning-rate schedule
   * Link: [torch.optim.lr_scheduler.CosineAnnealingLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
   * Date Accessed: 5 November 2026
* Automatic Mixed Precision:
   * Author: PyTorch Contributors, Linux Foundation
   * Used for: Mixed-precision training on the Colab GPU
   * Link: [Automatic Mixed Precision package](https://docs.pytorch.org/docs/stable/amp.html)
   * Date Accessed: 5 November 2026
* Reproducibility in PyTorch:
   * Author: PyTorch Contributors, Linux Foundation
   * Used for: Seeding, deterministic cuDNN settings and data loader seeding
   * Link: [Reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
   * Date Accessed: 5 November 2026
* Cross-Validation for Grouped Data:
   * Author: scikit-learn developers
   * Used for: Group-aware, stratified splitting so that near-duplicate images stay in one partition
   * Link: [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)
   * Date Accessed: 5 November 2026
* ROC AUC Score:
   * Author: scikit-learn developers
   * Used for: Computing AUC-ROC for validation, model selection and test evaluation
   * Link: [sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
   * Date Accessed: 5 November 2026
* McNemar Test:
   * Author: statsmodels developers
   * Used for: Paired comparison of accuracy, sensitivity and specificity, with the exact form for small discordant counts
   * Link: [statsmodels.stats.contingency_tables.mcnemar](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)
   * Date Accessed: 5 November 2026
* Wilson Confidence Intervals:
   * Author: statsmodels developers
   * Used for: 95% intervals for accuracy, sensitivity, specificity and precision
   * Link: [statsmodels.stats.proportion.proportion_confint](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html)
   * Date Accessed: 5 November 2026
* Fast DeLong Algorithm:
   * Author: Sun and Xu, IEEE Signal Processing Letters, 2014
   * Used for: The DeLong test comparing two correlated AUC-ROC values (H4)
   * Link: [Fast implementation of DeLong's algorithm](https://doi.org/10.1109/LSP.2014.2337313)
   * Date Accessed: 5 November 2026
* Grad-CAM:
   * Author: Selvaraju et al., International Journal of Computer Vision, 2020
   * Used for: Gradient-weighted class activation maps for the ResNet-50 interpretability subset
   * Link: [Grad-CAM: visual explanations from deep networks](https://doi.org/10.1007/s11263-019-01228-7)
   * Date Accessed: 5 November 2026
* Annotation Parsing:
   * Author: Python Software Foundation
   * Used for: Reading the bounding-box XML annotation files
   * Link: [xml.etree.ElementTree](https://docs.python.org/3/library/xml.etree.elementtree.html)
   * Date Accessed: 5 November 2026
* Dataset Download Client:
   * Author: Kaggle
   * Used for: Downloading the public dataset into the Colab runtime
   * Link: [kagglehub](https://github.com/Kaggle/kagglehub)
   * Date Accessed: 5 November 2026
* Tuberculosis Image Dataset:
   * Author: Saife, 2020, Kaggle
   * Used for: All images and bounding-box annotations analysed in the study
   * Link: [Tuberculosis image dataset](https://www.kaggle.com/datasets/saife245/tuberculosis-image-datasets)
   * Date Accessed: 5 November 2026
* Google Colab Runtime:
   * Author: Google
   * Used for: The GPU compute environment for the full 120-run grid
   * Link: [Colaboratory frequently asked questions](https://research.google.com/colaboratory/faq.html)
   * Date Accessed: 5 November 2026

---

## 11. Data, ethics and responsible use

- The study is a secondary analysis of a public dataset and involved no new data collection from people. An ethical clearance waiver for secondary studies was granted by the institution before the work began.
- The institution that produced the dataset is deliberately not named, in line with the anonymity condition of the waiver.
- The images are not redistributed in this repository. Obtain them from the original Kaggle source under its terms.
- Outputs are research findings about architecture choice. They must not be used to diagnose, triage or treat patients.

---

## 12. Project links and how to cite

### 12.1 Where to find the project

| Resource | What it holds | Link |
|---|---|---|
| GitHub repository | Analysis notebook, configuration, saved tables and this README | [![GitHub](https://img.shields.io/badge/Open-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/st10251759/RPDA8412---Research-Project---notebook) |
| Google Drive folder | The working folder in which the notebook was run in Google Colab, with its saved outputs | [![Google Drive](https://img.shields.io/badge/Open-Google%20Drive-4285F4?style=flat-square&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1_cEhWtv40TbBO1G7IbB0Fi_NK6wHljKa?usp=sharing) |

The two locations hold the same project. The GitHub repository is the version-controlled copy of the code, and the Google Drive folder is the environment the reported results were produced in. If the Drive folder does not open, check that you are signed in to a Google account.

### 12.2 Suggested citation

Harvard (Anglia) style:

> Chetty, C., 2026. *Research notebook: ResNet-50 versus ViT-Base for tuberculosis bacilli detection in Ziehl-Neelsen-stained sputum smears.* [Jupyter notebook] Available at: <https://github.com/st10251759/RPDA8412---Research-Project---notebook> [Accessed 5 November 2026].

BibTeX:

```bibtex
@misc{chetty2026notebook,
  author       = {Chetty, Cameron},
  title        = {Research notebook: ResNet-50 versus ViT-Base for tuberculosis bacilli detection in Ziehl-Neelsen-stained sputum smears},
  year         = {2026},
  howpublished = {\url{https://github.com/st10251759/RPDA8412---Research-Project---notebook}},
  note         = {Jupyter notebook. Accessed 5 November 2026}
}
```

Please also cite the dataset: Saife, M.A., 2020. *Tuberculosis image dataset.* [dataset] Kaggle. Available at: <https://www.kaggle.com/datasets/saife245/tuberculosis-image-datasets>.

### 12.3 Author and academic context

| | |
|---|---|
| Author | Cameron Chetty (ST10251759) |
| Supervisor | Dr Janine Lee Upton |
| Programme | Postgraduate Diploma in Data Analytics (NQF Level 8) |
| Institution | The Independent Institute of Education |
| Module | RPDA8412 Research Project |
