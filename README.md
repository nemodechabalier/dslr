# Data Science × Logistic Regression
## Harry Potter and the Data Scientist

A machine learning project implementing **one-vs-all logistic regression** to classify Hogwarts house assignments based on student performance data.

---

## 🎯 Objectives

- **Data Analysis**: Compute statistical properties (mean, std, min, max, percentiles) without built-in functions
- **Data Visualization**: Create histograms, scatter plots, and pair plots to explore relationships
- **Logistic Regression**: Train a multi-class classifier using three gradient descent variants:
  - Batch Gradient Descent
  - Stochastic Gradient Descent  
  - Mini-batch Gradient Descent
- **Evaluation**: Measure model performance on a validation split or on the provided test set when needed

---

## 📋 Setup

### 1. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify installation

```bash
python3 -c "import pandas, numpy, matplotlib, seaborn, sklearn; print('All imports successful')"
```

---

## 📁 Project Structure

```
dslr/
├── datasets/
│   ├── dataset_train.csv          # Training data with Hogwarts House labels
│   ├── dataset_test.csv           # Test data without labels (for submission)
│   └── logreg_weights.json        # Trained model weights
├── srcs/
│   ├── main/                      # Entry point scripts
│   │   ├── describe.py            # Dataset statistics
│   │   ├── histogram.py           # Feature distribution analysis
│   │   ├── scatter_plot.py        # Pairwise feature correlation
│   │   ├── pair_plot.py           # Full scatter plot matrix
│   │   ├── logreg_train.py        # Train classifier and save weights
│   │   ├── logreg_predict.py      # Make predictions on test data
│   │   └── logreg_validate.py     # Validate on held-out split
│   ├── data/                      # Data handling
│   │   ├── io.py                  # CSV loading
│   │   ├── preprocess.py          # Cleaning & normalization
│   │   ├── stats.py               # Statistical calculations
│   │   ├── models.py              # DatasetStore class
│   │   ├── pipeline.py            # Data pipeline orchestration
│   │   └── exceptions.py          # Custom exceptions
│   ├── ml/                        # Machine learning
│   │   ├── train.py               # Gradient descent variants
│   │   ├── predict.py             # Prediction logic
│   │   ├── utils.py               # Sigmoid, cost functions
│   │   └── __init__.py
│   └── visualization/             # Plotting functions
│       ├── histogram.py
│       ├── scatter_plot.py
│       ├── pair_plot.py
│       └── __init__.py
├── visu/                          # Output visualizations
│   ├── histograms/
│   ├── pair_plots/
│   └── scatter_plots/
├── houses.csv                     # Predictions output
├── requirements.txt
└── README.md
```

---

## 🚀 Usage

### Data Analysis

Display statistical summaries for all numerical features:

```bash
python3 srcs/main/describe.py datasets/dataset_train.csv
```

Output includes: Count, Mean, Std, Min, 25%, 50%, 75%, Max

---

### Data Visualization

**Histogram** — Identify homogeneous score distributions:

```bash
python3 srcs/main/histogram.py datasets/dataset_train.csv
```

**Scatter Plot** — Find correlated features:

```bash
python3 srcs/main/scatter_plot.py datasets/dataset_train.csv
```

**Pair Plot** — Visualize all feature relationships:

```bash
python3 srcs/main/pair_plot.py datasets/dataset_train.csv
```

---

### Logistic Regression

#### Training

Train a classifier and save weights to `datasets/logreg_weights.json`:

```bash
# Using batch gradient descent (10,000 iterations)
python3 srcs/main/logreg_train.py batch datasets/dataset_train.csv

# Using stochastic gradient descent (100 epochs)
python3 srcs/main/logreg_train.py stochastic datasets/dataset_train.csv

# Using mini-batch gradient descent (100 epochs, batch size 32)
python3 srcs/main/logreg_train.py mini-batch datasets/dataset_train.csv

# With custom features (default uses all 8 features)
python3 srcs/main/logreg_train.py batch datasets/dataset_train.csv Astronomy Charms Flying
```

#### Prediction

Generate predictions on test data:

```bash
python3 srcs/main/logreg_predict.py datasets/dataset_test.csv datasets/logreg_weights.json
```

Output: `houses.csv` with predictions for each test sample

#### Evaluation

You can evaluate the model on a held-out subset of the training data or on the provided test set using a custom script or notebook. This repository does not include a dedicated `logreg_validate.py` script.

A typical workflow is:

```bash
# 1. Train the model
python3 srcs/main/logreg_train.py batch datasets/dataset_train.csv

# 2. Generate predictions for the test set
python3 srcs/main/logreg_predict.py datasets/dataset_test.csv datasets/logreg_weights.json
```

The goal remains to reach the best possible accuracy on the evaluation data.

---

## 🔧 Gradient Descent Methods

| Method | Iterations | Data per Step | Convergence | Notes |
|--------|-----------|----------------|------------|-------|
| **Batch** | 10,000 | All samples | Smooth | Full gradient at each step |
| **Stochastic** | 100 epochs | 1 random sample | Noisy | High variance, escapes local minima |
| **Mini-batch** | 100 epochs | 32 samples | Balanced | Good compromise between batch & SGD |

Learning rate (α) = 0.1 for all methods

---

## 📊 Key Features

- **No heavy-lifting functions** — Statistics computed from scratch using NumPy
- **One-vs-all classification** — Trains 4 binary classifiers (one per Hogwarts house)
- **Data normalization** — Features standardized using training set statistics
- **Optional validation workflow** — You can evaluate on a held-out split or test set with your own script
- **Configurable features** — Select any subset of numerical features for training

---

## 🧪 Example Workflow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Explore data
python3 srcs/main/describe.py datasets/dataset_train.csv
python3 srcs/main/pair_plot.py datasets/dataset_train.csv

# 3. Train and compare different methods
python3 srcs/main/logreg_train.py batch datasets/dataset_train.csv
python3 srcs/main/logreg_train.py mini-batch datasets/dataset_train.csv

# 4. Generate predictions
python3 srcs/main/logreg_predict.py datasets/dataset_test.csv datasets/logreg_weights.json

# 5. Submit houses.csv
```

---

## 📈 Performance

- Target accuracy: **≥98%** on test data
- Optimization opportunities:
  - Feature selection & engineering
  - Learning rate tuning
  - Iteration/epoch count adjustment
  - Regularization (bonus)

---

## 📚 Mathematics Reference

### Cost Function (Binary Cross-Entropy)

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(h_\theta(x_i)) + (1-y_i) \log(1-h_\theta(x_i)) \right]$$

### Gradient Descent Update

$$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

### Sigmoid Function

$$h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$$

---

## 🔐 Dependencies

- **pandas** ≥2.0.0 — Data loading and manipulation
- **numpy** ≥1.24.0 — Numerical computations
- **matplotlib** ≥3.7.0 — Plotting
- **seaborn** ≥0.12.0 — Statistical visualization
- **scikit-learn** ≥1.3.0 — Validation metrics

---

## 👤 Author

Data Scientist defending Hogwarts (muggle edition)