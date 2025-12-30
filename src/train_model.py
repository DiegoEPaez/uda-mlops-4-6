# Script to train machine learning model.

# Add the necessary imports for the starter code.
import pandas as pd
from sklearn.model_selection import train_test_split

from .ml.data import process_data
from .ml.model import compute_metrics_slices, save_model, train_model

# Add code to load in the data.
data = pd.read_csv("data/census.csv")
data.columns = data.columns.str.strip()

# Drop some columns which are redundant or will not be used
# fnlwgt = final weight (how many people are in this category)
# is dropped since it is not useful for inference
data = data.drop(["fnlgt", "education-num"], axis=1)

# Optional: use K-fold CV instead of a single train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
model = train_model(X_train, y_train)
save_model(model, encoder, lb)

# Calulate performance on slices
sliced_metrics = compute_metrics_slices(test, model, cat_features, encoder, lb)
sliced_metrics.to_csv("model/slice_output.txt", index=False)
