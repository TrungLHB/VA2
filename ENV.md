# Environment Setup

## 1. Python Environment

Use Python 3.10 or newer.

Preferred setup with `uv`:

```bash
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

Fallback setup with standard `venv` and `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision pillow tensorboard opencv-python
```

## 2. Folder Layout

The project root should look like this:

```text
AV2/
├── 3DResNet/
│   ├── dataset.py
│   ├── main.py
│   └── model.py
├── TSN/
│   ├── dataset.py
│   ├── main.py
│   └── model.py
├── data/
│   ├── classes.txt
│   ├── train.txt
│   ├── validation.txt
│   ├── mini_UCF/
│   ├── mini_UCF_flow/
│   └── mini_UCF_frames/
├── pyproject.toml
├── uv.lock
└── ENV.md
```

`mini_UCF_frames/` contains extracted RGB frames. If it is missing or incomplete,
the dataset code can extract frames from the AVI files in `mini_UCF/` using
OpenCV.

## 3. Unzip the Data Folder

Place the provided dataset zip file in the project root.  The filename may vary,
for example `data.zip`, `miniUCF.zip`, or `miniUCF_data.zip`.

Unzip it into the project root:

```bash
unzip data.zip
```

If the zip extracts to a nested folder, move or rename it so the final path is:

```text
AV2/data/
```

The required files are:

```text
data/classes.txt
data/train.txt
data/validation.txt
data/mini_UCF/
data/mini_UCF_flow/
```

The raw RGB videos should be stored like this:

```text
data/mini_UCF/BalanceBeam/v_BalanceBeam_g02_c01.avi
```

The optical-flow files should be stored like this:

```text
data/mini_UCF_flow/BalanceBeam/v_BalanceBeam_g02_c01/flow_x_0001.jpg
data/mini_UCF_flow/BalanceBeam/v_BalanceBeam_g02_c01/flow_y_0001.jpg
```

## 4. Extract RGB Frames

RGB frames are extracted automatically when a dataset sees missing frames.  To
extract them explicitly, run either task dataset extractor:

```bash
python - <<'PY'
from TSN.dataset import extract_rgb_frames

extract_rgb_frames()
PY
```

This creates:

```text
data/mini_UCF_frames/
```

with frame files like:

```text
data/mini_UCF_frames/BalanceBeam/v_BalanceBeam_g02_c01/img_00001.jpg
```

## 5. Run Training
(Example parameters for Colab)

Train TSN with RGB input:

```bash
MODALITY=rgb USE_IMAGENET_INIT=1 EPOCHS=15 BATCH_SIZE=128 python ./TSN/main.py
```

Train TSN with optical flow:

```bash
MODALITY=flow USE_IMAGENET_INIT=0 EPOCHS=1 BATCH_SIZE=64 python ./TSN/main.py

MODALITY=flow USE_IMAGENET_INIT=1 EPOCHS=1 BATCH_SIZE=64 python ./TSN/main.py
```

Evaluate TSN late fusion after training the RGB ImageNet and flow ImageNet models:

```bash
MODALITY=fusion BATCH_SIZE=128 python ./TSN/main.py
```

By default, this loads:

```text
TSN/checkpoints/rgb_imagenet.pt
TSN/checkpoints/flow_imagenet.pt
```

Train 3D ResNet:

```bash
USE_IMAGENET_INIT=0 EPOCHS=5 BATCH_SIZE=128 python /content/VA2/3DResNet/main.py

USE_IMAGENET_INIT=1 EPOCHS=5 BATCH_SIZE=128 python /content/VA2/3DResNet/main.py
```

Checkpoints and TensorBoard logs are written under each task folder:

```text
TSN/checkpoints/
TSN/runs/
3DResNet/checkpoints/
3DResNet/runs/
```
