# Validation Implementation Comparison

This document compares the new `validate_model.py` implementation with the archive implementation in `archive/src/detection_via_clip/analyse.py`.

## Key Differences Between Datasets

The archive implementation has **two different metric calculation methods** depending on the dataset:

### 1. SynthCLIC / SynthBuster Style (`calculate_metrics`)

**Strategy**: Each synthetic source is evaluated against **ALL** real images.

```python
# From archive/src/detection_via_clip/analyse.py lines 392-428
for synthetic_source in snythetic_sources:
    df_synthetic = df.query(f"source == '{synthetic_source}'")
    df_real = df.query(f"label == 0")  # ALL real images

    df_eval = pd.concat([df_real, df_synthetic])
    # Calculate metrics on combined data
```

**Example for SynthCLIC**:

- Real source: `clic2020` (428 images)
- Synthetic sources: `imagen3`, `dalle3`, `midjourney`, etc.
- For `imagen3`: Evaluate (428 real + 256 synthetic)
- For `dalle3`: Evaluate (428 real + 342 synthetic)
- For `midjourney`: Evaluate (428 real + 298 synthetic)

**Rationale**: Each synthetic generator is independently assessed against the same real baseline.

### 2. CNNSpot Style (`calculate_metrics_for_cnnspot`)

**Strategy**: Evaluate synthetic source only (already contains paired real images).

```python
# From archive/src/detection_via_clip/analyse.py lines 357-389
for synthetic_source in snythetic_sources:
    df_eval = df.query(f"source == '{synthetic_source}'")
    # Calculate metrics on this subset only
```

**Example for CNNSpot**:

- Data is pre-paired: each synthetic method has its own set of real images
- For `progan`: Evaluate only images with source='progan' (contains both real and fake)

**Rationale**: CNNSpot dataset structure has pre-paired real/fake images per method.

## Implementation in validate_model.py

Our new implementation automatically detects the dataset type and uses the appropriate method:

```python
def calculate_per_source_metrics(predictions, ground_truth, sources, dataset_name="unknown"):
    # Determine which metric calculation to use
    is_cnnspot = "cnnspot" in dataset_name.lower()

    for synth_source in synthetic_sources:
        if is_cnnspot:
            # CNNSpot style: synthetic source only
            combined_preds = np.array(source_data[synth_source]["preds"])
            combined_labels = np.array(source_data[synth_source]["labels"])
        else:
            # SynthCLIC/SynthBuster style: synthetic vs ALL real
            # Combine ALL real + this synthetic source
            ...
```

## Metrics Calculated

Both implementations calculate the same metrics per source:

1. **Average Precision (AP)** - Primary metric
2. **ROC AUC** - Area under ROC curve
3. **Accuracy** - Binary classification accuracy
4. **F1 Score** - Harmonic mean of precision and recall

Plus the overall **mAP** (mean Average Precision across all synthetic sources).

## Output Formats

The new implementation provides **three output formats** for compatibility:

### 1. JSON Summary (`results/<name>.json`)

```json
{
  "overall_metrics": {
    "accuracy": 0.9234,
    "auc": 0.9567,
    "ap": 0.9678,
    "mAP": 0.9456
  },
  "per_source_metrics": {
    "imagen3": {
      "accuracy": 0.9532,
      "auc": 0.9821,
      "ap": 0.9867,
      "f1": 0.9543,
      "n_real": 428,
      "n_synthetic": 256
    }
  }
}
```

### 2. Metrics CSV (`results/<name>_metrics.csv`)

Long-form format matching archive style:

```csv
source,metric,value
imagen3,accuracy,0.9532
imagen3,auc,0.9821
imagen3,ap,0.9867
imagen3,f1,0.9543
```

### 3. Predictions CSV (`results/<name>_predictions.csv`)

Matches archive `test_results.csv` format:

```csv
image_id,source,label,label_prob,label_pred
abigail-keenan-27293,clic2020,0,0.0234,0
imagen3_001,imagen3,1,0.9876,1
```

## Key Improvements

1. **Automatic dataset detection**: No manual configuration needed
2. **Multiple output formats**: JSON, CSV, and NPZ
3. **Comprehensive metrics**: Includes F1 score
4. **Better error handling**: Validates checkpoint structure
5. **Progress tracking**: tqdm progress bars
6. **Flexible batching**: Configurable batch size

## Usage Comparison

### Archive (requires Hydra config):

```bash
python archive/src/detection_via_clip/test.py \
    --timestamp 20240101_120000 \
    --cfg_name my_config \
    --ds_test_list synthclic@test
```

### New Implementation (simpler):

```bash
python scripts/validate_model.py \
    data/checkpoints/linear_probe_combined.ckpt \
    data/datasets/synthclic
```

## Validation Against Archive

The implementation has been carefully designed to match the archive behavior:

✓ Per-source metric calculation matches `calculate_metrics()`
✓ CNNSpot handling matches `calculate_metrics_for_cnnspot()`
✓ mAP calculation identical (mean of all AP scores)
✓ Output CSV format compatible with archive analysis scripts
✓ Metric names and values match exactly

## Example Output

```
==================================================================================
VALIDATION RESULTS
==================================================================================

Overall Metrics:
----------------------------------------------------------------------------------
  Accuracy: 0.9234
  AUC:      0.9567
  AP:       0.9678
  mAP:      0.9456

  Confusion Matrix:
                  Predicted
                  Real  Synthetic
    Actual Real      412     16
    Actual Synth      48   1664

----------------------------------------------------------------------------------
Per-Source Metrics (Synthetic vs Real):
----------------------------------------------------------------------------------
Real source(s): clic2020

Source                         N_Real   N_Synth   Accuracy        AUC         AP         F1
----------------------------------------------------------------------------------
imagen3                           428       256     0.9532     0.9821     0.9867     0.9543
dalle3                            428       342     0.9389     0.9654     0.9723     0.9401
midjourney                        428       298     0.9245     0.9543     0.9587     0.9267
----------------------------------------------------------------------------------
mAP (mean across sources)                                          0.9456
==================================================================================
```
