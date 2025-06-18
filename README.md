# 🛡️ Diamond: Error-Controlled Interaction Discovery

**Diamond** is a Python library for discovering feature interactions in machine learning models **with rigorous false discovery rate (FDR) control**. It is particularly useful for scientific and biomedical applications where interaction interpretability and statistical reliability are essential.

---

## 🔍 Features

- **FDR-Controlled Interaction Discovery**  
  Uses the model-X knockoffs framework to ensure statistically valid interaction findings.

- **Non-Additivity Distillation**  
  Isolates interaction-specific importance beyond additive effects.

- **Model-Agnostic**  
  Compatible with various machine learning models, including Random Forests, XGBoost, and Neural Networks.

---

## ▶️ Quick Start

### 1. Clone the repository and install dependencies
```bash
git clone https://github.com/batmen-lab/diamond.git
cd diamond
conda env create -f environment.yml
conda activate diamond
# You also need to install the latest version of `xlearn` from the source code. Please follow the instructions in the [xlearn repository](https://github.com/aksnzhy/xlearn).
```

### 2. Prepare example data
```bash
unzip data.zip
# Ensure the unzipped `data/` folder is at the root level alongside `src/`
```

### 3. Run a demo

Example jupyter notebooks are provided in the `example` directory. You can run the notebooks to see how to use the **DIAMOND**.

---

## 📂 Repository Structure

| Path               | Description                                |
|--------------------|--------------------------------------------|
| `src/`             | Core implementation of the Diamond method  |
| `example/`         | Jupyter notebooks demonstrating usage      |
| `data.zip`         | Compressed example datasets                |
| `environment.yml`  | Conda environment specification            |
| `README.md`        | Project overview and usage instructions    |

---

## ❓ Why Use Diamond?

- Isolates **non-additive** interactions often missed by standard approaches
- Provides **FDR-controlled** interaction discovery via knockoff-based inference
- Supports diverse machine learning models for flexible use across domains

---

## 📝 Citation

If you use Diamond in your research, please cite:

> Chen W, Jiang Y, Noble WS, Lu YY. *Error-controlled non-additive interaction discovery in machine learning models.* *Nat Mach Intell* (Accepted, 2025).

---

## 🛠️ License

This project is licensed under the [Apache 2.0 License](LICENSE).

