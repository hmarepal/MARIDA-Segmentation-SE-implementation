# Marine Debris Segmentation with MARIDA

Semantic segmentation of marine debris and other ocean-surface classes in multispectral Sentinel-2 imagery using the MARIDA dataset.

This project compares a baseline U-Net against attention-based variants to study how spatial attention and Squeeze-and-Excitation (SE) channel attention affect segmentation performance, especially for the highly underrepresented **Marine Debris** class.

## Project Overview

Marine debris detection from satellite imagery is challenging because the target class occupies a very small portion of the labeled pixels and can be difficult to distinguish from visually similar ocean-surface materials.

This project uses:

- 11-band Sentinel-2 image patches
- 11 semantic segmentation classes
- Weighted cross-entropy loss to address class imbalance
- Data augmentation with random rotations and horizontal flips
- U-Net as the baseline architecture
- Attention gates for spatial attention
- Squeeze-and-Excitation blocks for channel attention

The experiments compare four architectures:

1. Plain U-Net
2. Attention U-Net
3. Attention U-Net + SE
4. Attention U-Net + Deep SE

## Dataset

The project uses the **MARIDA** marine debris dataset.

Dataset split used in the notebook:

| Split | Patches |
|---|---:|
| Train | 694 |
| Validation | 328 |
| Test | 359 |

Each input contains **11 spectral bands**.

The segmentation task contains the following 11 classes:

1. Marine Debris
2. Dense Sargassum
3. Sparse Sargassum
4. Natural Organic Material
5. Ship
6. Clouds
7. Marine Water
8. Sediment-Laden Water
9. Foam
10. Turbid Water
11. Shallow Water

### Class Imbalance

Marine Debris is strongly underrepresented in the dataset.

| Split | Marine Debris Pixels |
|---|---:|
| Train | 0.4525% |
| Validation | 0.5045% |
| Test | 0.1955% |

Because of this imbalance, overall pixel accuracy alone is not enough to evaluate performance. Mean IoU, Macro F1, and Marine Debris IoU are especially important in this project.

## Preprocessing

The notebook performs the following preprocessing steps:

- Reads multispectral `.tif` images with Rasterio
- Normalizes each spectral band using precomputed mean and standard deviation values
- Maps MARIDA labels into 11 segmentation classes
- Applies random 90-degree rotations during training
- Applies random horizontal flips during training
- Uses weighted cross-entropy loss to increase the contribution of rare classes

## Models

### 1. Plain U-Net

The baseline model uses a four-stage encoder-decoder architecture.

Encoder channels:

```text
11 -> 16 -> 32 -> 64 -> 128
```

Bottleneck:

```text
128 -> 256
```

The decoder uses transposed convolutions and standard U-Net skip connections.

### 2. Attention U-Net

The baseline U-Net is modified by adding attention gates to the skip connections.

These gates allow the decoder to selectively emphasize spatial features from the encoder before they are concatenated with the decoder feature maps.

### 3. Attention U-Net + SE

Squeeze-and-Excitation blocks are added to introduce channel attention.

The SE blocks learn channel-wise importance weights and rescale feature maps based on the global information contained in each channel.

### 4. Attention U-Net + Deep SE

The final experiment applies SE attention only to deeper feature representations rather than throughout the entire encoder.

SE blocks are applied to:

- Encoder stage 3
- Encoder stage 4
- Bottleneck

The decoder continues to use spatial attention gates.

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 2e-4 |
| Epochs | 45 |
| Batch Size | 5 |
| Input Bands | 11 |
| Classes | 11 |
| Loss | Weighted Cross Entropy |
| Random Seed | 42 |
| Mixed Precision | PyTorch AMP |
| Hardware Used | NVIDIA Tesla T4 |

The baseline U-Net also uses a MultiStepLR learning-rate scheduler with a milestone at epoch 40.

## Results

Test-set results:

| Model | Mean IoU | Macro F1 | Pixel Accuracy | Marine Debris IoU |
|---|---:|---:|---:|---:|
| Plain U-Net | 0.5842 | 0.7002 | 0.9297 | 0.1787 |
| Attention U-Net | 0.5869 | 0.7014 | 0.9308 | **0.2428** |
| Attention U-Net + SE | 0.5373 | 0.6396 | **0.9376** | 0.0828 |
| Attention U-Net + Deep SE | **0.5913** | **0.7075** | 0.9206 | 0.1839 |

## Key Findings

The experiments showed that adding attention did not produce a uniform improvement across every metric.

The **Attention U-Net** produced the strongest Marine Debris segmentation performance, increasing Marine Debris IoU from **0.1787 to 0.2428** compared with the baseline U-Net.

The **Deep-SE Attention U-Net** achieved the highest overall Mean IoU at **0.5913** and the highest Macro F1 at **0.7075**.

Applying SE attention more broadly throughout the model did not improve performance. The full-SE variant produced a lower Mean IoU of **0.5373** and Marine Debris IoU of **0.0828**.

This suggests that attention placement matters. Adding more attention modules does not automatically improve segmentation quality, and deeper channel attention performed better than applying SE blocks throughout the architecture.

## Repository Structure

```text
marine-debris-segmentation/
|
|-- marine-debris-segmentation.ipynb
|-- README.md
|-- requirements.txt
|-- LICENSE
|-- .gitignore
`-- results/
```

## Running the Notebook

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The notebook was developed in a Kaggle environment and expects the MARIDA dataset to be available under `/kaggle/input`.

The notebook searches recursively for the MARIDA `patches` directory and expects the dataset structure to include:

```text
MARIDA/
|-- patches/
`-- splits/
    |-- train_X.txt
    |-- val_X.txt
    `-- test_X.txt
```

If running locally, update the dataset path in the notebook to point to your local MARIDA directory.

## Requirements

Core dependencies include:

- Python
- PyTorch
- NumPy
- Rasterio
- Matplotlib
- Pandas

See `requirements.txt` for the installable package list.

## Future Work

Possible future directions include:

- Further investigation of spatial and channel attention placement
- Transformer or CNN-Transformer hybrid architectures
- Alternative loss functions for severe class imbalance
- Improved augmentation strategies
- Additional analysis focused specifically on Marine Debris precision and recall

## License

This project is released under the MIT License.

## Author

Hemal Marepalli
