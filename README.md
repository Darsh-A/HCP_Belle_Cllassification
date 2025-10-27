# Event Classification for HEP

This repo holds the code for the Data Science (IDC409 (Intro to DS and ML)) project of Group 23 
**Authors:**

Darsh A (MS22068)

Swalih (MS22132)

Rida Haris (MS22023)

Abhiram Srinivas (MS22093)


---

## Table of Contents

- [Data](#data)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Codebase Guide](#codebase-guide)
- [Results](#results)
- [Additional Resources](#additional-resources)

---

## Data

### Event Type Flags

$e^+ e^- \\to \\Upsilon (4S) \\to B^+ B^-$ = 0

$e^+ e^- \\to \\Upsilon(4S) \\to B^0 \\bar{B}^0$ = 1

$e^+ e^- \\to c \\bar{c}$ = 2

$e^+ e^- \\to u \\bar{u}$ = 3

$e^+ e^- \\to d \\bar{d}$ = 4

$e^+ e^- \\to s \\bar{s}$ = 5

**Binary Classification:** Add (0 and 1) to one class and (2, 3, 4, and 5) to other class

---

## Installation

### Linux (WSL Tested)

1. **Clone the repository**
```bash
git clone https://github.com/Darsh-A/IDC409-Classification-of-Events.git
```

2. **Run the setup script**
```bash
chmod +x install.sh
./install.sh
```

### Manual Installation

If any errors occur please run the commands manually after fixing the machine specific errors

The required python packages can be installed as:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** FastBDT needs to build on a linux system, the Code base was tested and worked on on WSL

---

## Project Structure

The code is setup as follow:

- **`/src`** - All the model files and utility files are stored here
- **`showcase.ipynb`** - Showcase notebook demonstrating the models
- **`/data`** - The data folder holds the event type data used for this project. To generate data please install basf2 (guide given below) and then follow this [guide](https://training.belle2.org/online_book/basf2/cs.html?utm_=)
- **`/models`** - The trained models are saved here
- **`/plots`** - Generated plots and visualizations

---

## Codebase Guide

### Model Structure

- All the models follow a similar structure for uniformity
- All models consists of a tune function that finds the best hyperparameter for the model, we have already trained them using RandomSearch and GridSearch on our data and saved it in `data/retraining_runtime_vs_performance.csv`

### Model Output Format

All models output the following JSON structure:
```python
result = {
    'confusion_matrix': cm,
    'roc_auc_score': roc_auc_score,
    'roc_curve': roc_curve,
    'accuracy': accuracy,
    'model': model,
    'feature_importance': feature_importance,
    'reduced_features': reduced_features,
    'training_time': end_time - start_time
}
```

---

## Results

We present a summary of the results of our project.

With the chosen emsemble of different Tree models we find out `XGBoost` performs _consistently_ better then all with gpu-acceoleration)

### ROC Comparison of Data Reduced Models

<img width="702" height="547" alt="image" src="https://github.com/user-attachments/assets/fdc0ead6-d081-471c-9252-99adf47eb87f" />

### Metrics for Best Performing Model (XGBoost)

<img width="1790" height="1398" alt="image" src="https://github.com/user-attachments/assets/ef0f1fb6-8274-429a-812f-b0013a2d8155" />

---

## Additional Resources

Optional Thing for the project and arent required, just for reference

### Basf2 Setup Guide

#### Initial Setup

##### 1. Launch WSL
```bash
wsl <path-to-project-directory>
```

##### 2. Clone Tools Repository
```bash
git clone https://github.com/belle2/tools.git
```

##### 3. Configure Environment

Add to `.bashrc`:
```bash
export PATH=/mnt/a/DSci/Projects/HEP_Event_Classf/tools:$PATH
```

Source the configuration:
```bash
source ~/.bashrc
```

##### 4. Set Permissions
```bash
chmod +x /mnt/a/DSci/Projects/HEP_Event_Classf/tools/b2*
```

##### 5. Initialize Basf2
```bash
source b2setup
```

##### 6. Install Release (~2GB)
```bash
b2install-release
```

**If SSH error occurs:**
```bash
git config --global url."https://github.com/".insteadOf git@github.com:
```

Install specific release:
```bash
b2install-release <version>  # Example: b2install-release 09-00-04
```

**If installation fails, install dependencies:**
```bash
sudo apt install scons gfortran python3-dev
```

##### 7. Install Externals (~6.5GB)

Check required version from error message in `b2setup release-<version>`, then:
```bash
b2install-externals <version>  # Example: b2install-externals 02-02-04
```

##### 8. Verify Installation
```bash
basf2 --info
```

#### Running Basf2

##### 1. Start WSL Session

Open WSL in your project directory (VS Code or terminal)

##### 2. Activate Environment
```bash
source /mnt/a/DSci/Projects/HEP_Event_Classf/tools/b2setup
b2setup release-09-00-04
```

##### 3. Execute Scripts
```bash
basf2 <script.py>  # Example: basf2 myscript.py
```

#### Quick Reference

| Command | Purpose |
|---------|---------|
| `b2install-release` | List/install Basf2 releases |
| `b2install-externals` | Install dependencies |
| `b2setup <release>` | Activate specific release |
| `basf2 --info` | Check installation |

#### Notes

- Replace `/mnt/a/DSci/Projects/HEP_Event_Classf/` with your actual project path
- Total disk space required: ~8.5GB (2GB release + 6.5GB externals)
- Setup needs to be done only once; afterwards just source and activate the environment

---
