## Optimal Embodiment

### Overview

This repository contains the code for the **Optimal Embodiment** project, focused on processing and analyzing human motion based on the AMASS dataset.

### Requirements

- **Python**: recommended >= 3.10  
- **Conda** (Anaconda or Miniconda) for environment management

### Environment setup with Conda

```bash
# Clone the repository (if you have not already)
git clone <URL_OF_THIS_REPOSITORY>
cd OptimalEmbodiment

# Create environment
conda create -n optimal-embodiment python=3.10 -y

# Activate environment
conda activate optimal-embodiment

# Install the package in editable mode
pip install -e .
```

### Downloading the AMASS dataset

1. Follow the instructions in the **official BlenderProc AMASS guide**:  
   [AMASS Human Poses Dataset – BlenderProc](https://dlr-rm.github.io/BlenderProc/examples/datasets/amass_human_poses/README.html).

2. Once the data is downloaded, create a `data/` directory in the root of the repository (if it does not already exist):

```bash
mkdir -p data
```

3. Copy the downloaded AMASS folders into `data/`, **keeping their original folder names**, for example:

```text
OptimalEmbodiment/
  data/
    AMASS/
      CMU/
      MPI_HDM05/
      ...
```

   The exact structure may vary depending on how you download/extract the dataset, but the idea is to preserve the original AMASS folder names.

### AMASS dataset preprocessing

To obtain the final dataset used by the main algorithm, you should implement a script `scripts/process_amass_data.py` in this repository.

Example usage (once implemented):

```bash
python scripts/process_amass_data.py \
  --input-dir data/AMASS \
  --output-dir data/processed_amass
```

- **`--input-dir`**: path to the root folder containing the original AMASS data (inside `data/`).  
- **`--output-dir`**: path where the processed dataset ready for the algorithm will be stored.

The exact preprocessing steps (normalization, joint selection, output format, feature computation, etc.) should be implemented inside `scripts/process_amass_data.py` according to the needs of your experiments.

### Citation

If you use this project or the AMASS dataset in scientific work, please cite at least:

**AMASS: Archive of Motion Capture as Surface Shapes**

```bibtex
@inproceedings{mahmood2019amass,
  title     = {AMASS: Archive of Motion Capture as Surface Shapes},
  author    = {Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F and Pons-Moll, Gerard and Black, Michael J},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2019}
}
```

Additionally, please cite any other relevant tools or libraries you use (for example, BlenderProc) following the recommendations of their authors.
