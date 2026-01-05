# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model in this project classifies individuals according to whether their annual income
is less or more than $50k. It is a Random Forest classifier trained on several demographic
attributes from Census Bureau data.

## Intended Use

The model is intended to predict the income bracket (>50k or ≤50k) when demographic
information about a person is available. Potential use cases include credit risk modeling,
fraud detection, business intelligence, and targeted marketing.

## Training Data

The training data is sourced from the U.S. Census Bureau (commonly the Adult Income
dataset). Predictive features include: workclass, education, marital-status, occupation,
relationship, race, sex, and native-country.

## Evaluation Data

The evaluation data is a held-out portion of the same Census Bureau dataset, using the
identical feature set as the training data.

## Metrics

The model was evaluated using the following classification metrics:

Precision: 0.70
Recall: 0.61
F1 Score: 0.65

Note: The feature fnlwgt (final weight) was dropped as it is unlikely to be available or
useful at inference time. Retaining it improves the F1 score slightly (to ~0.68), but it
was removed to ensure realistic deployment performance.

## Ethical Considerations

The are several slices that exhibit a low performance and should be further investigated:

variable, slice, precision, recall, f1_score
workclass, ?,0.4782608695652174,0.2972972972972973,0.36666666666666664
education, 7th-8th,0.5,0.16666666666666666,0.25
education, 10th,0.75,0.2727272727272727,0.4
education, 11th,0.6666666666666666,0.3333333333333333,0.4444444444444444
education, 12th,1.0,0.2,0.3333333333333333
marital-status, widowed,1.0,0.3076923076923077,0.47058823529411764
marital-status, separated,1.0,0.14285714285714285,0.25
relationship, Unmarried,0.8,0.21621621621621623,0.3404255319148936

## Caveats and Recommendations

Use the model with caution in high-stakes decisions due to observed performance disparities
across demographic groups. Additional bias mitigation, more representative training data, or
slice-specific thresholds are recommended before production deployment.
