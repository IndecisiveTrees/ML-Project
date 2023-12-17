import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import set_config
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    PowerTransformer,
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from xgboost import XGBClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from icd9cms.icd9 import search as icdsearch

sns.set_theme(style="white", palette="viridis")
pal = sns.color_palette("viridis")

pd.set_option("display.max_rows", 100)
set_config(transform_output="pandas")
pd.options.mode.chained_assignment = None

import gc
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENC_CAT_COLS = [
    "race",
    "gender",
    "age",
    "weight",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
]
ENC_CAT_COLS_NO_DRUGS = [
    "race",
    "gender",
    "age",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "change",
    "diabetesMed",
]
CAT_COLS = [
    "enc_id",
    "patient_id",
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "readmission_id",
]
CAT_COLS_WITH_DIAG = [
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "readmission_id",
]


def diag(df):
    # Initialize empty lists to store transformed values
    diag_1 = []
    diag_2 = []
    diag_3 = []

    # Iterate through the DataFrame to process the 'diag' columns
    for idx, row in df.iterrows():
        # Extract the values from 'diag_1', 'diag_2', and 'diag_3' columns
        d1 = str(row["diag_1"])
        d2 = str(row["diag_2"])
        d3 = str(row["diag_3"])

        # Handle missing values (NaN)
        if d1 == "nan":
            diag_1.append(np.nan)
        else:
            # Process ICD9 codes that start with 'E'
            if d1[0] == "E":
                d1 = d1[:4]
            # Truncate ICD9 codes to the first 3 characters for grouping
            elif len(d1) > 3:
                d1 = d1[:3]
            # Ensure a consistent format for ICD9 codes
            v1 = f"{int(d1):03d}" if d1.isnumeric() else d1
            # Use a function 'icdsearch' (not shown in this code) to obtain a parent node for the ICD9 code
            node = icdsearch(v1)
            if not node:
                print(v1, idx)
                break
            diag_1.append(str(node.parent))

        # Repeat the same process for 'diag_2' and 'diag_3'
        d2 = str(row["diag_2"])
        if d2 == "nan":
            diag_2.append(np.nan)
        else:
            if d2[0 == "E"]:
                d2 = d2[:4]
            elif len(d2) > 3:
                d2 = d2[:3]
            v2 = f"{int(d2):03d}" if d2.isnumeric() else d2
            node = icdsearch(v2)
            if not node:
                print(v2, idx)
                break
            diag_2.append(str(node.parent))
        d3 = str(row["diag_3"])
        if d3 == "nan":
            diag_3.append(np.nan)
        else:
            if d3[0 == "E"]:
                d3 = d3[:4]
            elif len(d3) > 3:
                d3 = d3[:3]
            v3 = f"{int(d3):03d}" if d3.isnumeric() else d3
            node = icdsearch(v3)
            if not node:
                print(v3, idx)
                break
            diag_3.append(str(node.parent))

    # Update the DataFrame with the transformed 'diag' columns
    df["diag_1"] = diag_1
    df["diag_2"] = diag_2
    df["diag_3"] = diag_3

    return df


def drugs(data, tt):
    # Define a nested function 'drug_changes' to process drug columns
    def drug_changes(row):
        d = {"drug_up": 0, "drug_down": 0, "drug_steady": 0}
        for drug in row:
            if drug == "Up":
                d["drug_up"] += 1
            elif drug == "Down":
                d["drug_down"] += 1
            elif drug == "Steady":
                d["drug_steady"] += 1
        return pd.Series(d)

    # Define a list of drug columns to be processed
    drugs = [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]

    # Extract only the relevant drug columns from the data
    drugs_data = data[drugs]

    # Apply the 'drug_changes' function to create a DataFrame with drug change information
    drug_change_df = drugs_data.apply(drug_changes, 1)

    # Remove the processed drug columns from the data
    data.drop(columns=drugs, inplace=True)

    # Handle data for training and testing separately
    if tt == "train":
        # Extract the target variable 'readmission_id' for training data
        y = data["readmission_id"]
        data.drop(columns=["readmission_id"], inplace=True)

        # Join the processed drug change information with the training data
        data = data.join(drug_change_df)
        data = data.join(y)
    elif tt == "test":
        # Join the processed drug change information with the test data
        data = data.join(drug_change_df)

    return data


def encode_cat(data, tt, cat_cols, enc):
    if tt == "train":
        # Encode categorical variables for the training data
        encoded_data = pd.DataFrame(enc.fit_transform(data[cat_cols]), columns=cat_cols)
        for col in cat_cols:
            data[col] = encoded_data[col]
    elif tt == "test":
        # Encode categorical variables for the test data using the same encoder
        encoded_data = pd.DataFrame(enc.transform(data[cat_cols]), columns=cat_cols)
        for col in cat_cols:
            data[col] = encoded_data[col]
    return data


def compute_pat_cnt(data, test_data, tt, scaler):
    # Calculate the count of admissions/re-admissions for each patient
    vc = pd.concat([data["patient_id"], test_data["patient_id"]], axis=0).value_counts()
    pat_cnt = []
    for idx, row in data.iterrows():
        pat_cnt.append(vc[row["patient_id"]])

    if tt == "train":
        # Insert the 'pat_cnt' feature before the last column in the training data
        data.insert(data.shape[1] - 1, "pat_cnt", pat_cnt)
        # Standardize the 'pat_cnt' feature using scaler
        data["pat_cnt"] = scaler.fit_transform(
            data["pat_cnt"].to_numpy().reshape(-1, 1)
        )
    elif tt == "test":
        # Insert the 'pat_cnt' feature at the end of the test data
        data.insert(data.shape[1], "pat_cnt", pat_cnt)
        data["pat_cnt"] = scaler.transform(data["pat_cnt"].to_numpy().reshape(-1, 1))

    return data


def removing_null(data, tt, imputer):
    # Drop columns with significant null values
    data.drop(columns=["weight", "max_glu_serum", "A1Cresult"], inplace=True)

    # Impute constant values for 'medical_specialty' and 'payer_code'
    data["medical_specialty"] = data["medical_specialty"].fillna(68)
    data["payer_code"] = data["payer_code"].fillna(17)

    if tt == "train":
        # Extract the target variable 'readmission_id' for training data
        y = data["readmission_id"]

        # Remove the target variable column
        data = data.iloc[:, : data.shape[1] - 1]

        # Impute missing values for the remaining features using the provided imputer
        imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        # Rejoin the target variable for training data
        imputed_data = imputed_data.join(y)
    elif tt == "test":
        # Impute missing values for the test data using the same imputer
        imputed_data = pd.DataFrame(imputer.transform(data), columns=data.columns)

    return imputed_data


def deal_with_ids(data, tt, scaler):
    if tt == "train":
        # Normalize 'enc_id' for the training data using the provided scaler
        data["enc_id"] = scaler.fit_transform(data["enc_id"].to_numpy().reshape(-1, 1))
    elif tt == "test":
        # Normalize 'enc_id' for the test data using the same scaler
        data["enc_id"] = scaler.transform(data["enc_id"].to_numpy().reshape(-1, 1))

    return data


def get_X_y(data):
    # Extract features (X) and the target variable (y) from the given DataFrame
    X, y = data.iloc[:, : data.shape[1] - 1], data.iloc[:, data.shape[1] - 1]
    return X, y


def cv(model, X, y, params=None):
    # Perform cross-validation of the given model
    cv_results = cross_validate(
        model,
        X,
        y,
        scoring=["accuracy", "f1_macro"],
        return_estimator=True,
        fit_params=params,
    )
    return cv_results


def gen_submission(data, model, enc_ids, fname, xg=False, np=True):
    # Make predictions using the given model
    if not xg:
        if np:
            x = data
        else:
            x = data.to_numpy()
        preds = model.predict(x)
    else:
        x = xgb.DMatrix(data)
        preds = model.predict(x)

    if np:
        # Create a DataFrame for submission
        d = {"enc_id": enc_ids, "readmission_id": preds}
        submission = pd.DataFrame(d)
    else:
        # Update the 'readmission_id' column with predictions
        data["readmission_id"] = preds
        data["enc_id"] = enc_ids
        submission = data[["enc_id", "readmission_id"]]

    # Ensure data types of 'enc_id' and 'readmission_id'
    submission.loc[:, "enc_id"] = submission["enc_id"].astype(int)
    submission.loc[:, "readmission_id"] = submission["readmission_id"].astype(float)

    # Save the submission DataFrame to a CSV file with the provided filename
    submission.to_csv(fname, index=False)


def load_data(data_dir):
    # Load the training and test data from the specified directory
    data = pd.read_csv(data_dir + "/train.csv")
    test_data = pd.read_csv(data_dir + "/test.csv")
    return data, test_data


def preprocessing_and_fe(data, test_data):
    # Extract 'enc_id' from test data for submission
    enc_ids = test_data["enc_id"]

    # Process the 'diag' columns for both training and test data
    data = diag(data)
    test_data = diag(test_data)

    # Process the 'drugs' columns for training and test data
    data = drugs(data, "train")
    test_data = drugs(test_data, "test")

    # Encode categorical variables using OrdinalEncoder for training and test data
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    data = encode_cat(data, "train", ENC_CAT_COLS_NO_DRUGS, enc)
    test_data = encode_cat(test_data, "test", ENC_CAT_COLS_NO_DRUGS, enc)

    # Standardize the 'pat_cnt' feature for training and test data
    ss1 = StandardScaler()
    data = compute_pat_cnt(data, test_data, "train", ss1)
    test_data = compute_pat_cnt(test_data, data, "test", ss1)

    # Impute missing values and handle null values for training and test data
    imputer = SimpleImputer(strategy="most_frequent")
    data = removing_null(data, "train", imputer)
    test_data = removing_null(test_data, "test", imputer)

    # Standardize the 'enc_id' feature for training and test data
    ss2 = StandardScaler()
    data = deal_with_ids(data, "train", ss2)
    test_data = deal_with_ids(test_data, "test", ss2)

    # Create a list of categorical features based on the presence of 'diag' columns
    cat_feat = list(map(lambda x: x in CAT_COLS_WITH_DIAG, data.columns.tolist()[:-1]))

    # Extract X (input features) and y (target variable) for training data
    X, y = get_X_y(data)

    # Convert test data to a NumPy array
    x = test_data

    return X, y, x, enc_ids, cat_feat
