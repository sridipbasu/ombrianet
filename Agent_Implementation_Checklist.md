# Agent Implementation Checklist

This document is designed for an AI agent to fill in while implementing and testing the paper-matching checklist.

## Metadata

```yaml
project: OMBRIA
paper_summary: OmbriaNetSupervised.md
source_repo: OMBRIA-master
status_legend:
  TODO: not started
  DOING: in progress
  DONE: implemented and verified
  FAIL: implemented but failing tests
  NA: not applicable
```

## Change Log

| Date | Phase | Summary | Status | Notes |
| --- | --- | --- | --- | --- |
| YYYY-MM-DD | Phase X | | TODO | |

## Implementation Checklist

Fill the table below as you implement items. Use one of: TODO, DOING, DONE, FAIL, NA.

| ID | Phase | Requirement | Paper Reference | Repo Evidence | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| DS-01 | Dataset | Events and dates match paper table | Table I | | TODO | |
| DS-02 | Dataset | Sentinel-1 GRD VV + 30x30m median filter | Dataset section | | TODO | |
| DS-03 | Dataset | Sentinel-2 bands B3/B8/B11 | Dataset section | | TODO | |
| DS-04 | Dataset | 256x256 non-overlapping tiles | Dataset section | | TODO | |
| DS-05 | Dataset | Timestamp windows (May pre-event, event to +15d) | Dataset section | | TODO | |
| DS-06 | Dataset | Reproject to UTM zone per event | Dataset section | | TODO | |
| DS-07 | Dataset | Split 80/10/10 with shuffle | Dataset section | | TODO | |
| DS-08 | Dataset | Augmentation: flips, shifts, shear, rotations | Dataset section | | TODO | |
| UN-01 | U-Net | LeakyReLU activations | Method section | | TODO | |
| UN-02 | U-Net | 2x2 max pooling | Method section | | TODO | |
| UN-03 | U-Net | Upsampling + skip connections | Method section | | TODO | |
| UN-04 | U-Net | Final 1x1 conv + sigmoid | Method section | | TODO | |
| UN-05 | U-Net | Loss = binary cross-entropy | Method section | | TODO | |
| UN-06 | U-Net | Optimizer = Adam | Method section | | TODO | |
| UN-07 | U-Net | Post-event only input | Method section | | TODO | |
| UN-08 | U-Net | Epochs/batch sizes match paper | Experiments | | TODO | |
| BI-01 | Bitemporal | Two input streams (same modality) | Method section | | TODO | |
| BI-02 | Bitemporal | 3 conv blocks each stream | Method section | | TODO | |
| BI-03 | Bitemporal | Bottleneck concat | Method section | | TODO | |
| BI-04 | Bitemporal | Dropout = 0.3 | Method section | | TODO | |
| BI-05 | Bitemporal | Decoder with upsampling + skips | Method section | | TODO | |
| BI-06 | Bitemporal | Loss = binary cross-entropy | Method section | | TODO | |
| BI-07 | Bitemporal | Optimizer = Adam | Method section | | TODO | |
| BI-08 | Bitemporal | Param count ~10.8M | Method section | | TODO | |
| BI-09 | Bitemporal | Epochs/batch sizes match paper | Experiments | | TODO | |
| MM-01 | Multimodal | Four inputs (S2 pre/post, S1 pre/post) | Method section | | TODO | |
| MM-02 | Multimodal | 3 conv blocks each stream | Method section | | TODO | |
| MM-03 | Multimodal | Bottleneck concat | Method section | | TODO | |
| MM-04 | Multimodal | Dropout = 0.3 | Method section | | TODO | |
| MM-05 | Multimodal | Decoder with upsampling + skips | Method section | | TODO | |
| MM-06 | Multimodal | Loss = binary cross-entropy | Method section | | TODO | |
| MM-07 | Multimodal | Optimizer = Adam | Method section | | TODO | |
| MM-08 | Multimodal | Param count ~18.1M | Method section | | TODO | |
| MM-09 | Multimodal | Epochs/batch sizes match paper | Experiments | | TODO | |
| BL-01 | Baselines | Otsu + MNDWI from B3/B11 | Experiments | | TODO | |
| BL-02 | Baselines | SVM linear, C=10, raw pixels | Experiments | | TODO | |
| EV-01 | Evaluation | PA, IoU, FW IoU definitions match | Metrics section | | TODO | |
| EV-02 | Evaluation | Test split matches paper | Experiments | | TODO | |
| EV-03 | Evaluation | Results match paper tables | Results section | | TODO | |
| RP-01 | Repro | Set random seeds | Repro section | | TODO | |
| RP-02 | Repro | Save checkpoints + history | Repro section | | TODO | |
| RP-03 | Repro | Capture pip freeze | Repro section | | TODO | |

## Test Plan and Results

Fill these rows as tests are implemented and executed.

| Test ID | Description | Data Subset | Expected | Actual | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| T-DS-01 | Tile size and channels check | small sample | 256x256, S1=1 ch, S2=3 ch | | TODO | |
| T-DS-02 | Split ratio check | full index | 80/10/10 within tolerance | | TODO | |
| T-MD-01 | U-Net param count | model build | matches paper | | TODO | |
| T-MD-02 | Bitemporal param count | model build | matches paper | | TODO | |
| T-MD-03 | Multimodal param count | model build | matches paper | | TODO | |
| T-OPT-01 | Optimizer and loss check | model config | Adam + BCE | | TODO | |
| T-BASE-01 | Otsu/MNDWI output sanity | small sample | produces binary mask | | TODO | |
| T-BASE-02 | SVM config check | small sample | LinearSVC C=10 | | TODO | |
| T-EVAL-01 | Metric correctness on toy input | unit test | expected values | | TODO | |

## Notes

- Update Repo Evidence with file paths or commits as work progresses.
- Keep Status consistent and only mark DONE after tests pass.
