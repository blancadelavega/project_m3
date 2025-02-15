# Diamond Price Prediction Challenge

This project is part of Module 3 of the Iron Hack Bootcamp and focuses on solving the **Diamond Price Prediction Challenge**. The goal is to build an end-to-end machine learning workflow to predict the price of diamonds based on various features.

The evaluation metric chosen for this competition is **RMSE (Root Mean Squared Error)**.

---

## Workflow Overview

### 1. Exploratory Data Analysis (EDA)
- **Objective:** Understand data distribution, identify missing or outlier values, and explore relationships between variables.
- Two analyses were performed: a basic and an in-depth analysis, found in the `Basic_EDA` and `In-depth_EDA` notebooks.

### 2. Feature Engineering
- Derived variables were created to improve predictive performance:
  - **`volume`**: Computed as the product of dimensions `x`, `y`, and `z`.
  - **`density`**: Computed as the ratio of `carat` to `volume`.
- **Label Encoding** was applied to categorical variables: `cut`, `color`, and `clarity`.

### 3. Modeling and Best Model Selection: *PrimeModel*
The modeling process was carried out in the `PrimeModel` notebook and followed these steps:

1. **Data Loading:**
   - Training data was loaded from `diamonds_train.csv`.

2. **Feature Engineering:**
   - Computed `volume` and `density` while handling zero values to avoid division errors.

3. **Encoding Categorical Variables:**
   - Applied `LabelEncoder` to transform `cut`, `color`, and `clarity` into numerical format.

4. **Feature and Target Selection:**
   - **Features:** `carat`, `depth`, `table`, `density`, `cut`, `color`, and `clarity`.
   - **Target:** `price`.

5. **Model Training and Hyperparameter Optimization:**
   - A `RandomForestRegressor` model was trained using `GridSearchCV` to find the best hyperparameters.
   - Hyperparameters tested:
     - `n_estimators`: [100, 200, 300]
     - `max_depth`: [None, 3, 10]
     - `min_samples_split`: [2, 10]
     - `min_samples_leaf`: [1, 4]
     - `max_features`: [None, 'sqrt', 'log2']
   - **Results:**
     - **Best Hyperparameters:** `{'max_depth': None, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}`
     - **Best RMSE:** ~550.11 (using `neg_root_mean_squared_error` in GridSearchCV).

6. **Model Persistence:**
   - The optimized model was saved as `.pkl` using `joblib` for future use.

7. **Prediction and Submission Generation:**
   - The same preprocessing steps were applied to the test set (`diamonds_test.csv`).
   - Predictions were made using the saved model.
   - Results were stored in a CSV file (`PrimeModel.csv`) inside `data/sample_submissions/`.

---

## Requirements and Execution

### Requirements
- **Python 3.x**
- Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `joblib`

### Execution
1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. **Install Dependencies:**

   If using `pip`:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you create a `requirements.txt` file listing the required libraries, if not already available.)*

3. **Run the Notebooks:**

   - Open the development environment (e.g., [Visual Studio Code](https://code.visualstudio.com/) or Jupyter Notebook).
   - Run the notebooks in the `notebooks` directory sequentially, starting with `Basic_EDA` to understand the data, followed by the modeling notebooks.
   - The `PrimeModel` notebook contains the full pipeline for the final model.

4. **Generate Submission:**
   - Once the model is trained and predictions are made on the test set, the `PrimeModel.csv` file will be saved in `data/sample_submissions/`.
   - This file is formatted correctly for submission.

---