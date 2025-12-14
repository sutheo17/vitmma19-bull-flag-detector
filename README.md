# Project Details

## Data Preparation & Setup
The data pipeline supports a hybrid approach, allowing for both fully automated ingestion and manual data injection for custom testing or inference.

Directory Structure & Docker -> The system relies on two local folders, Data and Output, mounted to the container at runtime:

* `/app/data` (Mapped to local Data): The workspace for datasets (raw and processed).

* `/app/output` (Mapped to local Output): Confusion matrices, evaluation metrics and inference results get exported here.

There are two ways to add training / inference data:

1. Automated Ingestion -> **!! By default !!**, `01-data-preprocessing.py` automates the process:

Fetches the primary dataset ZIP from the configured URL (url is in `config.py`, it downloads my dataset from `OneDrive Repository/bullflagdetector/XOBJYX/data.zip`) and extracts it into /app/data. This automatically populates the training data folders (e.g., XOBJYX, J2QIYD...) and also provides the inference folder containing sample CSVs, allowing for immediate prediction testing without any manual setup.

You can toggle this on / off using the `DOWNLOAD_FROM_ONEDRIVE` variable inside `config.py`.

2. Manual Data Injection -> Users can manually extend / add the dataset or run predictions by adding files directly to the local Data folder:

**Training/Test Data:** To add labeled data, place a new subfolder inside Data containing your CSV files and a corresponding .json label file. The preprocessing script automatically parses these inputs, extracting specific labeled flag instances from the CSVs based on the JSON timestamps - handling cases where a single CSV file contains multiple distinct flags - and integrates them into the training set.

**Inference Data:** To predict on new assets, simply drop raw CSV files into the Data/inference folder. The `04-inference.py` script detects these files, processes them, and outputs confidence scores without requiring retraining.

## Solution Description

### Problem 
The objective was to classify chart patterns in financial time-series data into six distinct categories: Bullish and Bearish variants of Normal Flags, Pennants, and Wedges. Did not implemented real-time detection of these flags, so my task was classifying pre-segmented OHLC data windows.

### Model Architecture 
A custom 1D Convolutional Neural Network (CNN) was chosen to effectively extract local temporal features from the price sequences. The architecture consists of:

- Feature Extractor: Three stacked convolutional blocks, each containing a 1D convolution layer, Batch Normalization, ReLU activation, Max Pooling, and Dropout for regularization.

- Classifier: A fully connected (dense) layer that maps the flattened feature maps to the six output classes.

### Training Methodology 
The training pipeline focused on robustness and generalization:

- Preprocessing: Sequences were normalized relative to the "pole" start to focus on percentage changes rather than absolute price, and interpolated to a fixed length of 80 steps.

- Imbalance Handling: Inverse class weighting was applied to the Loss function to prevent the model from biasing toward the majority class (Normal Flags).

- Optimization: The model was trained using the Adam optimizer with a learning rate scheduler (ReduceLROnPlateau) and Early Stopping to prevent overfitting. Data augmentation (scaling jitter) was applied during training to improve model invariance.

### Results 
The CNN model outperformed the rule-based heuristic baseline, however, given the limited dataset of only ~320 samples, expanding the data collection would be essential to further improve generalization and stability

## Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

### Run

To run the solution, use the following command. You must mount your local data directory to `/app/data` and your local output directory to `/app/output` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run --rm --gpus all -v /absolute/path/to/your/local/data:/app/data -v /absolute/path/to/your/local/output:/app/output dl-project > log/run.log 2>&1
```

*   Replace `/absolute/path/to/your/local/data` with the actual path to your dataset on your host machine that meets the [Data Preparation & Setup](#data-preparation--setup).
*   Replace `/absolute/path/to/your/local/output` with the actual path to your desired output folder on your host machine that meets the [Data Preparation & Setup](#data-preparation--setup).
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference, baseline comparison).

## File Structure and Functions

[Update according to the final file structure.]

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data. If set in `config.py`, it will download the training data and a few inference samples. The processed data is saved to `/app/data/processed_data.npz`.
    - `02-training.py`: The main script for defining the model and executing the training loop. It uses the processed data found inside `/app/data/processed_data.npz`. The trained model is saved into `/app/model.pth`.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics. The model is loaded from `/app/model.pth`. The confusion matrice (`confusion_matrix.png`) and evaluation metrics (`final_metrics.txt`) are saved to `/app/output`.
    - `04-inference.py`: Script for running the model on new, unseen data to generate predictions. The inference results are saved to `/app/output/inference_results.csv`. The csv file contains the following columns: [file,prediction,confidence]
    - `05-baseline-comparison.py`: Script that implements a rule-based heuristic baseline (using linear regression slopes) and benchmarks it against the trained deep learning model to quantify performance improvements. It reads the input CSV files from the inference directory `/app/inference`. The confusion matrice (`baseline_confusion_matrix.png`) and evaluation metrics (`baseline_metrics.txt`) are saved to `/app/output`.
    - `config.py`: Configuration file containing hyperparameters, paths and behaviour variables.
    - `run.sh`: Shell script executing the full pipeline. Default command executed when the container starts.
    - `utils.py`: Helper functions and utilities used across different scripts (logging).

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-data-exploration.ipynb`: Notebook for initial exploratory data analysis (EDA) and visualization.
    - `02-label-analysis.ipynb`: Notebook for analyzing the distribution and properties of the target labels.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
