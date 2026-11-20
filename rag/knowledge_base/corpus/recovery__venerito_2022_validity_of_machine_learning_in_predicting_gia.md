---
title: "Validity of Machine Learning in Predicting Giant Cell Arteritis Flare After Glucocorticoids Tapering"
authors: "Venerito V, Emmi G, Cantarini L, Leccese P, Fornaro M, Fabiani C, Lascaro N, Coladonato L, Mattioli I, Righetti G, Malandrino D, Tangaro S, Palermo A, Urban ML, Conticini E, Frediani B, Iannone F, Lopalco G"
journal: "Frontiers in immunology"
year: 2022
doi: "10.3389/fimmu.2022.860877"
pmcid: "PMC9017227"
license: "CC BY"
source_url: "https://europepmc.org/article/PMC/PMC9017227"
topic: recovery
citation: "Venerito V, Emmi G, Cantarini L, Leccese P, Fornaro M, Fabiani C, Lascaro N, Coladonato L, Mattioli I, Righetti G, Malandrino D, Tangaro S, Palermo A, Urban ML, Conticini E, Frediani B, Iannone F, Lopalco G (2022). Validity of Machine Learning in Predicting Giant Cell Arteritis Flare After Glucocorticoids Tapering. Frontiers in immunology."
---

# Validity of Machine Learning in Predicting Giant Cell Arteritis Flare After Glucocorticoids Tapering

**Source:** Venerito V, Emmi G, Cantarini L, Leccese P, Fornaro M, Fabiani C, Lascaro N, Coladonato L, Mattioli I, Righetti G, Malandrino D, Tangaro S, Palermo A, Urban ML, Conticini E, Frediani B, Iannone F, Lopalco G (2022). Validity of Machine Learning in Predicting Giant Cell Arteritis Flare After Glucocorticoids Tapering. Frontiers in immunology. DOI: 10.3389/fimmu.2022.860877. License: CC BY.

## Abstract

Inferential statistical methods failed in identifying reliable biomarkers and risk factors for relapsing giant cell arteritis (GCA) after glucocorticoids (GCs) tapering. A ML approach allows to handle complex non-linear relationships between patient attributes that are hard to model with traditional statistical methods, merging them to output a forecast or a probability for a given outcome.

The objective of the study was to assess whether ML algorithms can predict GCA relapse after GCs tapering.

GCA patients who underwent GCs therapy and regular follow-up visits for at least 12 months, were retrospectively analyzed and used for implementing 3 ML algorithms, namely, Logistic Regression (LR), Decision Tree (DT), and Random Forest (RF). The outcome of interest was disease relapse within 3 months during GCs tapering. After a ML variable selection method, based on a XGBoost wrapper, an attribute core set was used to train and test each algorithm using 5-fold cross-validation. The performance of each algorithm in both phases was assessed in terms of accuracy and area under receiver operating characteristic curve (AUROC).

The dataset consisted of 107 GCA patients (73 women, 68.2%) with mean age ( ± SD) 74.1 ( ± 8.5) years at presentation. GCA flare occurred in 40/107 patients (37.4%) within 3 months after GCs tapering. As a result of ML wrapper, the attribute core set with the least number of variables used for algorithm training included presence/absence of diabetes mellitus and concomitant polymyalgia rheumatica as well as erythrocyte sedimentation rate level at GCs baseline. RF showed the best performance, being significantly superior to other algorithms in accuracy (RF 71.4% vs LR 70.4% vs DT 62.9%). Consistently, RF precision (72.1%) was significantly greater than those of LR (62.6%) and DT (50.8%). Conversely, LR was superior to RF and DT in recall (RF 60% vs LR 62.5% vs DT 47.5%). Moreover, RF AUROC (0.76) was more significant compared to LR (0.73) and DT (0.65).

RF algorithm can predict GCA relapse after GCs tapering with sufficient accuracy. To date, this is one of the most accurate predictive modelings for such outcome. This ML method represents a reproducible tool, capable of supporting clinicians in GCA patient management.

## Introduction

Giant Cell Arteritis (GCA) is the most common systemic vasculitis worldwide (

In this regard, machine learning (ML) is emerging as a promising tool for implementing complex multi-parametric decision algorithms (

Here we investigate whether an explainable ML approach may be helpful to predict a GCA flare upon GC treatment tapering.


## Materials and Methods


### Data Gathering

Patients with classified GCA according to the 1990 ACR criteria ( NCT01543594 ). This study followed the STARD guidelines and the TRIPOD statement. All patients provided written informed consent.


### Outcome of Interest

The predictive modeling analysis aimed to forecast the probability of GCA flare at 3 months following GCs tapering, represented as a Boolean variable. Flare was considered as the recurrence of signs or symptoms of GCA. We kept track of the first flare following GCs tapering only.


### Attributes Selection

The attribute core set used to train the algorithms was determined using a cross-validated recursive feature elimination wrapper based on a decision tree algorithm with extreme gradient boosting (XGBoost) described elsewhere ( Supplementary Material Supplementary Material


### Algorithm’s Training and Validation

The analysis was implemented in a Python 3.8 environment using scikit-learn (ver. 0.22.1) and XGBoost (ver. 1.1.0) libraries ( Supplementary Material Supplementary Material


### Probability Calibration

A classification model generally forecasts a binary outcome for a given observation and class. In the process of predicting, a model may output the probability of an observation belonging to each possible class (


## Results

Our analysis included 107 patients with classified GCA (73 women, 68.2%) with mean age ( ± SD) of 74.1 ± 8.5 years and mean symptom duration of 17.7 ± 37.5 weeks who underwent GCs treatment with a mean dose of 27.7 ± 17.6 prednisone dose mg/daily. Headache was the most prevalent symptom at GCs baseline, being present in 82/107 patients (76.6%); 33 out of 107 (30.8%) complained of ophthalmologic symptoms, whereas PMR was diagnosed in 58 of them (54.2%). Temporal artery abnormalities were detected at the clinical exam in 33/105 patients (31.4%), in two patients, information about signs of superficial artery inflammation was missing. A “halo” sign was seen in 67 of the 78 patients (85.9%) who underwent temporal artery ultrasound examination. Temporal artery biopsy was carried out in 27 patients and was positive in 15 of them (55.6%). Mean Body Mass Index (BMI) was 26.1 ± 4.2, whereas 21/107 (19.6%) had diabetes mellitus. Mean ESR and CRP at GCs baseline was 70.6 ± 62.2 mm/h and 50.5 ± 80.8 mg/L, respectively ( Table 1 Table 1 Figure 1 Supplementary Material Figure 2 k -fold cross-validation produces a sampling from the error distribution of the model, it allows computing the expected value of the metrics and the standard deviation on the metrics. We ran paired two-tailed t-tests to confirm that the differences in results between different models were statistically significant. Each possible pairing of metrics always resulted in statistically significant differences between the models (p <0.0001 in all mentioned cases). For further details, see Table 2 Figure 3 Figure 4 Figure 1


## Discussion

Despite GCs treatment, several reports in the literature have shown that up to 68% of patients will have at least one relapse during GCA, thus increasing both the cumulative amount and duration of GCs therapy ( Figures 1 3


## Publisher’s Note

All claims expressed in this article are solely those of the authors and do not necessarily represent those of their affiliated organizations, or those of the publisher, the editors and the reviewers. Any product that may be evaluated in this article, or claim that may be made by its manufacturer, is not guaranteed or endorsed by the publisher.
