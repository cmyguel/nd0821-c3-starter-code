# Model Card

## Model Details
- Person developing the model: Cristian Del Toro, in the context of Udacity Course: "Machine Learning DevOps Engineer Nanodegree Program".
- Date: 02/2022
- Model type: Random Forest Classifier.

## Intended Use
- Prediction task is to determine whether a person makes over 50K a year.
- To be deployed with a RESTful API in Heroku, developed with Fast API.

## Training Data
Extraction was done by Barry Becker from the 1994 Census database. More information on the dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income).


## Evaluation Data
Training data and Evaluation data come from the exact same distribution. The split was done at random, no stratification was done.

## Metrics
- Computed metrics are precision, recall and fbeta.
- fbeta was the single evaluation metric used to select the model.
- General results of the model are (0.738, 0.645, 0.689) respectively.
- Training code generates the same metrics by slices of data for the categorical features.

## Ethical Considerations
Improper use of the data or model could lead to discrimination based on race, gender or country or origin.

## Caveats and Recommendations
Data Slicing suggest some categories with low performance (aside from 'unknown' tags) that may require further work and analysis for better performance.
