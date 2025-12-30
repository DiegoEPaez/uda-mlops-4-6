# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model contained in this project classify people according to their income whether they earn
less or more than 50k annually. A random forest based on several attributes achieves this
classification based on Census Bureau data.

## Intended Use

The model should be used to derive the attribute of income when the demographics of a person
are available. This can be useful in a variety of scenarios such as credit models, fraud prediction,
business intelligence, among others.

## Training Data

The data for training is obtained from the census bureau, and variables such as: workclass education, marital-status, occupation, relationship, race, sex, native-country, are used for
prediction.

## Evaluation Data

The data for training is obtained from the census bureau, and variables such as: workclass education, marital-status, occupation, relationship, race, sex, native-country, are used for
prediction.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

Overall the model exhibits these metrics:

Precision: 0.7,
Recall: 0.61,
F1 Score: 0.65

Please note that the variable fnlwgt (or final weight) was dropped since it is
most likely not useful for inference. However, this variable is useful for improving
the model performance (from 0.68 f1 score to 0.65), and is correlated with income.

## Ethical Considerations

The are several slices that exhibit a low performance and should be further investigated:


variable, slice, precision, recall, f1_score
workclass, ?,0.4782608695652174,0.2972972972972973,0.36666666666666664
education, 7th-8th,0.5,0.16666666666666666,0.25
education, 10th,0.75,0.2727272727272727,0.4
education, 11th,0.6666666666666666,0.3333333333333333,0.4444444444444444
education, 12th,1.0,0.2,0.3333333333333333
marital-status, Widowed,1.0,0.3076923076923077,0.47058823529411764
marital-status, Separated,1.0,0.14285714285714285,0.25
relationship, Unmarried,0.8,0.21621621621621623,0.3404255319148936

## Caveats and Recommendations

Use model as
