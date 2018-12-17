# Pediatric Sleep Apnea Prediction
Code for our upcoming paper on predicting pediatric obstructivecontains on sleep apnea (OSA).
Using relatively generic machine learning techniques, and straightforward feature engineering, we can outperform the nearest clinician baseline for risk-prioritizing children for full sleep study analysis.
We leverage a new and proprietary dataset for this use case.

## Overview
The package is roughly structured as:
    
    1. load_data.py           : tools for loading and cleaning the data
    2. feature_engineering.py : pipeline for handling & engineering data
    3. models.py              : set of models to be optimized
        3a. points_model.py   : Kang et al (2015) heuristic model
    4. evaluate.py            : loads data & models, and evaluates

## Modeling Challenges
Modeling OSA in this case is tough for several reasons.
First, __data is small__.
Our proprietary dataset contains only 456 patient observations.
Good modelling prevents _overfitting_ and, if possible, provides measures of _uncertainty_ and _interpretability_. 
Second, __data contains selection bias__.
The dataset contains data from children who were presented to a point-of-care with indicative symptoms.
This limits external validity; further testing ought to be done with a population-representative sample.
Third, __we limited the data we could use__.
We ignored first-level sleep study analysis metrics, as the goal of the paper is to prioritize children for a sleep study.
We are left with only 3 variables.

## Results
Coming very soon!

## TODO

- [ ] Check, re-factor eval pipeline
- [ ] Implement explicit model averaging
- [ ] Consider SMOTE / synthetic data for marginal performance boosts
- [ ] Find clinician heuristics newer than Kang et al. 2015
- [ ] Comment on statistical power given small data & cross-val optimization
