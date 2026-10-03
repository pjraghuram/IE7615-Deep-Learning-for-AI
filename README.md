# IE7615 Deep Learning for AI: CNN Object Classification

Compares simple CNNs trained from scratch with pretrained (transfer-learning) CNNs on 73 everyday objects photographed from different perspectives.

## Dataset
- `images_<OBJID>/<OBJID>_<sequence>.jpg`: one folder per object (`OBJ001` to `OBJ073`)
- `object_names.csv`: object ID → name (e.g. `OBJ001` → Basketball)
- 7,343 images; 7,315 after removing duplicates; RGB, 224x224

## Notebooks (run in order)
| Notebook | What it does | Output |
|---|---|---|
| `1_data_loading_and_preprocessing.ipynb` | Load, dedup (MD5), resize, augmentation, 70/15/15 split | `data/dataset.npz` |
| `2_simple_cnn.ipynb` | 4 simple CNNs (CNN_1 to CNN_4) | `models/CNN_*.keras`, `results/` |
| `3_pretrained_models.ipynb` | MobileNetV2, ResNet50V2, EfficientNetB0 (frozen + Dense head) | `models/*.keras`, `results/` |
| `4_test_and_results.ipynb` | Train/val/test evaluation, tables, plots, final model selection | `results/all_results.csv` |

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
In VS Code / Jupyter, select the `.venv` kernel.

## Running
1. Run notebook 1 to create `data/dataset.npz` (about 1 minute; not in the repo because of its size).
2. Get the trained models: either copy the shared `models/` folder into the project root, or retrain with notebooks 2 and 3 (about 3–4 hours on CPU).
3. Run notebook 4.

Random seed 42 is fixed, so the split and the results are reproducible.

## Results
| Model | Type | Params | Val Acc | Test Acc |
|---|---|---|---|---|
| CNN_1 [32, 64] | Simple CNN | 25.7M | 60.0% | 61.2% |
| CNN_2 [32, 64, 64] | Simple CNN | 6.5M | 63.0% | 63.0% |
| CNN_3 [32, 64, 128] | Simple CNN | 12.9M | 63.1% | 62.9% |
| CNN_4 [32, 64, 128, 128] | Simple CNN | 3.5M | 66.5% | 68.1% |
| MobileNetV2 | Pretrained | 2.6M | 88.3% | 89.8% |
| ResNet50V2 | Pretrained | 24.1M | 89.2% | 90.4% |
| **EfficientNetB0** | **Pretrained** | **4.4M** | **91.5%** | **92.4%** |

**Selected model: EfficientNetB0**: highest validation and test accuracy, smallest train−test gap and fastest training. See notebook 4, sections 4.9–4.10.
