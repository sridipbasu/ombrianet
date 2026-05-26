# Paper Matching Checklist

Status legend: [x] matches paper, [ ] does not match or not verified.

| Dataset Construction | U-Net Baseline | Bitemporal OmbriaNet | Multimodal OmbriaNet | Baselines | Evaluation and Metrics | Reproducibility | Current Repo Differences |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ ] Dataset events and dates match paper table. | [x] LeakyReLU activations. | [x] Two input streams, three conv blocks each. | [x] Four inputs (S2 pre/post, S1 pre/post). | [x] Otsu uses MNDWI from Sentinel-2. | [x] Compute PA, IoU, FW IoU. | [ ] Record random seeds. | [ ] U-Net uses RMSprop; should be Adam. |
| Notes: Not verified in repo. | Notes: Found in [UNET.ipynb](UNET.ipynb). | Notes: Found in [OmbriaNet.ipynb](OmbriaNet.ipynb). | Notes: Found in [OmbriaNet.ipynb](OmbriaNet.ipynb). | Notes: Found in [Otsu Threshold.ipynb](Otsu%20Threshold.ipynb). | Notes: Found in [Evaluation.ipynb](Evaluation.ipynb). | Notes: Not found in notebooks. | Notes: RMSprop in [UNET.ipynb](UNET.ipynb). |
| [ ] S1 GRD VV, 30x30m median filter. | [x] 2x2 max pooling, stride 2. | [x] Bottleneck feature concat. | [x] Bottleneck feature concat. | [x] SVM linear, C=10, raw pixels. | [ ] Same test split as paper. | [ ] Save checkpoints and history. | [ ] OmbriaNet dropout 0.2; should be 0.3. |
| Notes: Not verified in repo. | Notes: Found in [UNET.ipynb](UNET.ipynb). | Notes: Found in [OmbriaNet.ipynb](OmbriaNet.ipynb). | Notes: Found in [OmbriaNet.ipynb](OmbriaNet.ipynb). | Notes: Found in [SVM.ipynb](SVM.ipynb). | Notes: Split varies by generator. | Notes: Not consistently saved. | Notes: Dropout 0.2 in [OmbriaNet.ipynb](OmbriaNet.ipynb). |
| [ ] S2 bands B3, B8, B11 with stated handling. | [x] 2x2 upsampling and skip connections. | [ ] Dropout 0.3. | [ ] Dropout 0.3. | [ ] Results compared to paper tables. | [ ] Reported numbers align with paper. | [ ] Capture pip freeze. | [ ] GEE preprocessing pipeline missing. |
| Notes: Not verified in repo. | Notes: Found in [UNET.ipynb](UNET.ipynb). | Notes: Dropout 0.2 in [OmbriaNet.ipynb](OmbriaNet.ipynb). | Notes: Dropout 0.2 in [OmbriaNet.ipynb](OmbriaNet.ipynb). | Notes: Not verified in repo. | Notes: Not verified in repo. | Notes: Not in repo. | Notes: Not included in this repo. |
| [ ] Tiles are 256x256, non-overlapping. | [x] Final 1x1 conv with sigmoid. | [x] Decoder uses upsampling + skip connections. | [x] Decoder uses upsampling + skip connections. |  |  |  | [ ] Train/val split inconsistent; align to 80/10/10. |
| Notes: Not verified in repo. | Notes: Found in [UNET.ipynb](UNET.ipynb). | Notes: Found in [OmbriaNet.ipynb](OmbriaNet.ipynb). | Notes: Found in [OmbriaNet.ipynb](OmbriaNet.ipynb). |  |  |  | Notes: Validation split differs in generators. |
| [ ] Cloud filtering and timestamp windows. | [x] Loss is binary cross-entropy. | [x] Loss is binary cross-entropy. | [x] Loss is binary cross-entropy. |  |  |  |  |
| Notes: Not verified in repo. | Notes: Found in [UNET.ipynb](UNET.ipynb). | Notes: Found in [OmbriaNet.ipynb](OmbriaNet.ipynb). | Notes: Found in [OmbriaNet.ipynb](OmbriaNet.ipynb). |  |  |  |  |
| [ ] Reprojection to UTM per event. | [ ] Optimizer is Adam. | [x] Optimizer is Adam. | [x] Optimizer is Adam. |  |  |  |  |
| Notes: Not verified in repo. | Notes: RMSprop in [UNET.ipynb](UNET.ipynb). | Notes: Adam in [OmbriaNet.ipynb](OmbriaNet.ipynb). | Notes: Adam in [OmbriaNet.ipynb](OmbriaNet.ipynb). |  |  |  |  |
| [ ] Split is 80/10/10 with shuffling. | [ ] Input is post-event only. | [ ] Parameter count ~10.8M. | [ ] Parameter count ~18.1M and training settings match paper. |  |  |  |  |
| Notes: Validation split varies by generator. | Notes: Not verified in repo. | Notes: Not verified in repo. | Notes: Not verified in repo. |  |  |  |  |
| [x] Augmentation: flips, shifts, shear, rotations. | [ ] Training epochs and batch sizes match paper tables. | [ ] Training epochs and batch sizes match paper tables. |  |  |  |  |  |
| Notes: Present in [OmbriaNet.ipynb](OmbriaNet.ipynb). | Notes: Not verified in repo. | Notes: Not verified in repo. |  |  |  |  |  |
