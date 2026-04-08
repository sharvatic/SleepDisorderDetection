# Sleep Disorder Detection using Machine Learning

This project implements a machine learning pipeline to detect sleep disorders using physiological signals.

## Features

- **Data Preprocessing**: Cleans and preprocesses physiological data.
- **Feature Extraction**: Extracts relevant features for sleep disorder detection.
- **Model Training**: Trains machine learning models (e.g., SVM, Random Forest).
- **Evaluation**: Evaluates model performance using various metrics.

## Prerequisites

- Python 3.8+
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SleepDisorderDetection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to train and evaluate the model:

```bash
python main.py
```

## Project Structure

```
SleepDisorderDetection/
├── data/                  # Dataset files
├── models/                # Trained models
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── data_loader.py     # Data loading
│   ├── preprocessing.py   # Data preprocessing
│   ├── features.py        # Feature extraction
│   ├── train.py           # Model training
│   └── evaluate.py        # Model evaluation
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## Dataset

The dataset consists of physiological signals used for sleep disorder detection. For more information about the dataset, please refer to the original source.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.