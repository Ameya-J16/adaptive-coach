---
title: "Advanced feature engineering in Acute:Chronic Workload Ratio (ACWR) calculation for injury forecasting in elite soccer"
authors: "Matas-Bustos JB, Mora-García AM, de Hoyo Lora M, Nieto-Alarcón A, Gonzalez-Fernández FT"
journal: "PloS one"
year: 2025
doi: "10.1371/journal.pone.0327960"
pmcid: "PMC12286412"
license: "CC BY"
source_url: "https://europepmc.org/article/PMC/PMC12286412"
topic: training_load
citation: "Matas-Bustos JB, Mora-García AM, de Hoyo Lora M, Nieto-Alarcón A, Gonzalez-Fernández FT (2025). Advanced feature engineering in Acute:Chronic Workload Ratio (ACWR) calculation for injury forecasting in elite soccer. PloS one."
---

# Advanced feature engineering in Acute:Chronic Workload Ratio (ACWR) calculation for injury forecasting in elite soccer

**Source:** Matas-Bustos JB, Mora-García AM, de Hoyo Lora M, Nieto-Alarcón A, Gonzalez-Fernández FT (2025). Advanced feature engineering in Acute:Chronic Workload Ratio (ACWR) calculation for injury forecasting in elite soccer. PloS one. DOI: 10.1371/journal.pone.0327960. License: CC BY.

## Abstract

Controlling training monotony and monitoring external workload using the Acute:Chronic Workload Ratio (ACWR) is a common practice among elite soccer teams to prevent non-contact injuries. However, recent research has questioned whether ACWR offers sufficient predictive power for injury prevention in elite competition settings. In this paper, we propose a novel feature engineering framework for training load management, inspired by bilinear modeling and signal processing principles. Our method represents external workload variables, derived from GPS data, as discrete time series, which are then integrated into a temporal matrix termed the Footballer Workload Footprint (FWF). We introduce calculus-based techniques—applying integral and differential operations—to derive two representations from the FWF matrix: a cumulative workload matrix ( ∑ T F W F Δ T F W F

These findings indicate that our approach offers a robust, interpretable, and generalizable framework for sports science and medical professionals involved in injury prevention and training load monitoring.

## 1 Introduction

Injuries represent one of the most undesirable and unpredictable events that professional soccer players may experience over the course of a season. Soccer is a sport characterized by frequent physical contact; however, non-contact musculoskeletal injuries, such as hamstring strains and anterior cruciate ligament ruptures, remain common. Indeed, more than 90% of all muscle injuries and 51%–64% of joint and ligament injuries in soccer occur in non-contact situations [

In general, training load is strongly associated with injury and illness risk in athletes [ acute:chronic workload ratio (ACWR) for a set of variables [ sweet spot ”, the risk is evaluated [

This process has become an established and widely adopted practice in elite soccer teams to prevent non-contact injuries and minimize the risk [ traditional calculations of ACWR are ‘mathematically coupled’, as the most recent week is included in the estimation of both the acute and chronic workloads. ” [

In response, recent literature has proposed monitoring training progression by combining acute and chronic measures through a variety of strategies. For example, grouping time windows to measure week-to-week changes [

Should these methods be considered sufficient? Can analytical ACWR calculations truly predict injuries the day before a match, in the context of modern elite soccer? Artificial Intelligence/Machine Learning (AI/ML) has been used to identify athletes at high injury risk during sport participation, and that it may be helpful to identify injury risk factors [

In this study, we introduce a novel approach to model and manage player training workloads, aiming to improve the prediction of non-contact injuries in elite soccer, particularly in the period leading up to matches. We present a new approach to control training load inspired by bilinear modeling [

To assess our proposal, we compared the models trained using our novel approach with those trained using the most commonly referenced and utilized ACWR calculations by practitioners, specifically (‘ ACWR COUPLED ACWR UNCOUPLED ACWR EWMA

This study constitutes a first step toward evaluating the validity of the FWF method, using widely accepted supervised machine learning techniques to ensure fair and objective assessment of the proposed approach. The study will evaluate models using key performance metrics, such as accuracy, Type I and Type II errors, G-mean, also ROC/AUC and PR/RC curves. We expect our models to demonstrate strong predictive performance during testing on new data, particularly in distinguishing between injury and non-injury events, with balanced results across evaluation metrics. We will compare our results with the most cited studies in the field, to the best of our knowledge [

This paper is structured in five sections, including this introduction.


## 2 Materials and methods

The design and phases of our experiment can be reviewed in


### 2.1 Participants

A total of 23 professional soccer players voluntarily agreed to participate in the data collection for this study. The study was carried out according to the Declaration of Helsinki [

Accordingly, throughout the study, we refer to each soccer player as ‘ f k


### 2.2 Data collection and data extraction

The data used in this study were collected over the course of one full season, including both the preseason and the competitive period, from July to May. During this time, the team participated in two official domestic competitions—the Spanish league and the Spanish cup—as well as in a European tournament. The dataset was provided to the researchers in September 2023 to carry out the analyses described in this manuscript.

We analyzed a total of 4,124 samples (from both training sessions and matches) corresponding to 23 individual players. All official matches played were included in the analysis. Training data consisted exclusively of ‘on-pitch’ sessions scheduled by the coaching staff. Sessions such as individual training, recovery, and rehabilitation were excluded. When a player was injured, he was only re-included in the study after completing one full week of training with the team. Goalkeepers were excluded from the analysis.

We stand for each season event ‘ e j

Player movement during training sessions and matches was recorded using the WIMU PRO portable GPS device (a hybrid GNSS/LPS unit equipped with 4 x 3D accelerometers up to 1000 Hz, 3 x gyroscopes up to 1000 Hz, a 3D magnetometer at 100 Hz, and a barometer at 120 kPa, among other sensors). The GPS device was positioned between the shoulder blades using a compression vest to reduce movement artifacts. Speed and distance were sampled at 20 Hz, a frequency previously shown to offer valid and reliable measurements of instantaneous velocity across different movement intensities [

To ensure optimal performance, GPS units were activated and placed outdoors 15 minutes before warm-up [

We denote a collection of RAW data coming from GPS sensors for a footballer ‘ f k e j

where ‘ n ’ denotes the discrete sample index corresponding to the 20Hz sampling rate (i.e., one sample every Δ t = 0.05

The raw GPS data for each player ‘ f k e j V ’ of external workload variables extracted in an event ‘ e j s ’ sensors of GPS RAW ( n ) f k e j

where:

For this study, we extracted for every event ‘ e j f k V ’ of external workload variables.

The set of workload variables ‘ V ’ was constructed by extracting external workload variables from the raw signals acquired by the multiple sensors embedded in the GPS units, including GNSS positioning, tri-axial accelerometry, gyroscopy, magnetometry, and barometric pressure sensors. The set includes metrics such as distance, speed, acceleration, deceleration, step balance, player load, jump count, and impact forces. It also includes variables that describe the distribution of speed and acceleration across defined ranges, as well as the percentage of time spent at different intensities. Additional metrics capture the duration and intensity of activity, power of movement, number of sprints, and time-event-related data. Some variables were directly measured (e.g., distance covered, instantaneous speed, linear acceleration), while others were derived through processing and expert domain knowledge (e.g., playerload, number of sprints, step balance, or jump counts). A complete list and description of the 73 variables ‘ v i

The injury reports, provided by the club’s medical services, included complete information on all injuries sustained by players throughout the season and were fully anonymized. In total, 49 injuries were reported by the medical personnel and, in particular, 33 non-traumatic muscular injuries were recorded during the season. According to UEFA regulations, a non-contact injury is defined as any tissue damage that causes a player to be absent from physical activity for at least one day following the onset of symptoms.

Only injuries sustained by the 23 participants in this study were considered; all others were excluded from the analysis. In line with the objectives of our investigation and following recommendations from previous studies [

Leveraging the injury reports provided by the club, we have the capability to formulate the following linear application ‘ l ’ (

where, ‘ f k ∈ F e j ∈ E f k e j


### 2.3 Data modeling and preprocessing: Introducing the Footballer Workload Footprint (FWF)

The first major contribution of this paper lies in the domain of data modeling [

The primary goal of the proposed model is to improve daily monitoring of training monotony (TM) and to optimize the tracking of both internal and external workload variables in elite soccer players. As previously described, this problem is framed across three dimensions: F (footballers,

To handle this complexity, we represent the data following principles from data-centric process systems engineering [ variable-wise ’ bilinear approximation for batch processes [

Inspiring us also on theoretical bases of signal processing [ v i f k n ’ season events, as a Discrete Time Series (DTS) denoted as ‘ X v i ( n )

Afterward, we joined together all ‘ X v i ( n ) f k

where:

We named this temporal discrete matrix ( footprint ”, because it represents the external workload of the footballer as a “ footprint in the time ”. The proposed model is flexible and configurable, since it let us to determine the number of events and workload variables, and it allows to explore and focus the data in the most convenient temporal window required for any specific analysis.

Also, a compatible structure is necessary for managing injuries in accordance with ‘ FWF ’. We denote the matrix of injuries of a footballer ‘ f k n ’ season events, as a discrete time series (DTS) denoted as ‘ L f k ( n )

where:

This last structure complements the ‘ footprint ’, concluding our data modeling process and allowing us to manage information with agility in order to design and build the different datasets used in our experiments.

Beyond supporting our experimental framework, the Footballer Workload Footprint (FWF) also facilitates the standardized sharing of precise digital information about a footballer over a specific time span. We believe this approach could be seriously considered as a future industry standard by sports scientists and medical staff. It promotes collaboration with the data science community in studies involving workload datasets, as it provides a “ common language ” for structuring and analyzing data.

The conceptual foundation of the footprint lies in its ability to extract, transform, and load heterogeneous data sources—such as internal and external workload, musculoskeletal and biochemical data, nutritional and wellness factors—into a unified FWF representation for a given player ‘ f k footprints datasets ”—suitable for comparative and longitudinal analysis.

It also supports the integration of data from multiple teams, measurement systems, or contexts with limited injury records, enabling the construction of larger, more generalizable databases. This is particularly relevant in a domain where data collection protocols often differ, for example, due to variations in GPS hardware. Moreover, existing datasets tend to be highly imbalanced, with very few injury cases—a problem already noted in the introduction. Standardized and anonymized repositories of injured player footprints could therefore play a critical role in advancing injury modeling and predictive analytics in elite sport.


### 2.4 Exploratory and confirmatory analysis: Comparing the predictive effectiveness of ACWR against ‘ Cumulative and Temporal Variations of FWF ’

Does the calculation of ‘ Cumulative and Temporal variations of the FWF ’ matrix improve the predictive ability of the non-contact injury risk associated with the acute/chronic workload ratio (ACWR)? To evaluate the effectiveness of our proposed methodology, we utilized multivariate supervised techniques to perform confirmatory data analysis.

This analysis entailed comparing models trained using our innovative approach with those trained using the most commonly cited and used by practitioners, ACWR calculations: ‘ ACWR C O U P L E D ACWR UNCOUPLED ACWR E W M A

To perform all data preprocessing, feature engineering, model training, validation, and evaluation procedures, as describes

As depicted in ( V × E n ) ’ Eq ( f k L f k ( n ) FWF ’. Summarizing, the final designs of the five datasets were as follows:

Subsequently, as delineated in

The details for each of these phases are outlined below, providing a comprehensive justification and explanation for each phase:

After completion of the transformations for all datasets, the training/validation pipeline was initiated in corresponding ‘ X training


## 3 Results

In order to conduct the confirmatory analysis, we trained a total of 160 models using the eight multivariate supervised techniques described in

To complement the evaluation based on the best single model performance, we also analyzed the statistical distribution of the testing metrics across 20 independent repetitions for each trained algorithm. Specifically, we calculated the mean and the 95% confidence intervals (IC95%) for ROC-AUC, G-Mean, Accuracy, PR-AUC, Type I Error, and Type II Error. Confidence intervals were computed using the t-Student distribution, appropriate for the moderate sample size (n = 20), ensuring a more robust statistical estimation compared to classical normal assumptions. This approach strengthens the reliability of the comparative results across models. Consistent with standard practices in applied machine learning studies, model performance on the independent testing set was evaluated based on mean metrics and 95% confidence intervals (IC95%), without performing additional p-value calculations [

In order to assess the different datasets and identify the most effective models in our experimental development, we performed ROC/AUC and PR/AUC curve analyses, as described in

Additionally, to further validate the robustness and reliability of our findings, we performed permutation tests, as we detailed

Finally, in

These findings form the basis for the critical comparative analysis and interpretation discussed in the following section.


## 4 Discussion

Evaluating smart models for binary classification, as in our context of predicting non-contact injuries, presents difficulties in choosing the best model due to the necessity of balancing discriminative power along with sensitivity and specificity trade-offs.

Indicators, further discussed in (

Besides ROC-AUC and the G-mean index, the PR-AUC metric provides particular insights into a model’s ability to forecast positive instances and identify most positive cases in datasets with significant class imbalance [

In conclusion, selecting the optimal binary classification model in our scenario requires maximizing ROC-AUC, Gmean, and PR-AUC, while minimizing Type I and Type II Errors. Accuracy is the least reliable metric because a model can exhibit high accuracy alongside significant Type I or Type II errors.

Based on the stated criteria, we initially present a discussion of the model’s results following the training and validation process across the five datasets we designed. For a detailed analysis, we recommend referring to

A review of the average and variability of all trained models shows that Dataset 5 stands out as the best performer, achieving high ROC-AUC, G-mean, and PR-AUC scores, thereby balancing discriminative ability and fair classification, even though there are possible high Type II Error rates, particularly during validation.

In the context of our application, Type II Error evaluates whether we correctly predicted that a player would not get injured, but eventually did. It is crucial to select models that minimize this metric to avoid false negatives. Exposing a player to competitive stress in a high-level match when there is a significant risk of injury not only leads to the loss of the player for that particular match, but may also result in his unavailability for the following weeks during recovery and readjustment to competitive conditions [

This practice is already common in elite soccer, and our models trained with the cumulative and temporary variations of the FWF matrix ( ∑ T F W F V × E ( n ) f k Δ T F W F V × E ( n ) f k ACWR U N C O U P L E D ACWR C O U P L E D ACWR E W M A

In Dataset 2, the algorithm that performs the best during training is CART, which aligns with the findings of Rossi [

In Dataset 5, SVM emerges as the best-performing algorithm in both training and validation phases, significantly enhancing the results seen with CART in Dataset 2. For instance, the Gmean metric’s decline in validation is not observed with SVM in Dataset 5 (unlike with CART in Dataset 2), and during validation, the Type I (0.281±0.026) and Type II (0.333±0.119) Errors fall within ranges considered acceptable and very promising for real competition applications.

The findings suggest that linear and Bayesian classification models are inadequate for this issue, mainly due to their subpar training performance and poor validation efficacy. Conversely, we argue that Support Vector Machines (SVMs) are suitable for this scenario, given their effectiveness in managing nonlinear structures [

The main weakness of the results obtained by the models trained with SVM lies in the low PR-AUC, which is common to all the trained models and for all the datasets in this study. This implies that the models have difficulties to correctly identify the instances of the positive class (the injured), especially in unbalanced datasets, as is the case of the real-world environment of our problem. Let us recall that the objective of physical trainers and medical teams is precisely to avoid the events of the positive class, i.e., that soccer players get injured. The main issue is not the lack of injury samples, as these will naturally become more abundant over time, due to elite clubs allocating more resources in this direction. Rather, the key difficulty lies in the lack of an established industry standard for gathering, modeling, and processing players workloads and injury data for appropriate reuse in research and/or for creating global-shared, anonymized and public models that can be retrained and refined over time. This paper illustrates that our proposed method, the Footballer Workload Footprint (FWF), is efficient, practical, and resilient in tackling these challenges.

The results of the permutation tests described in p -value ( ≤ 0.001 ∑ T F W F Δ T F W F

A summary of the results of bank of player footprint’s ’ the event prior to the one in which the injury occurred. This repository could act as a reference and can be utilized by various teams and sports organizations. If we analyze the balance between Bias and Variance, the obtained bias (0.05247) is notably higher than the variance (0.01218), so the model has a moderate bias but a low variance. This could be interpreted as a model that is slightly underfitted, i.e., it could benefit from slight additional complexity to better capture the underlying function. However, the low variance indicates that the model is robust and consistent in its predictions. Consequently, there remains considerable potential for enhancing the classifier that has yet to be utilized. Given that the bias surpasses the variance, one might refine the model by fine-tuning the SVM hyperparameters, such as the regularization parameter C or the choice of kernel, to achieve a more optimal trade-off that reduces both bias and variance.

All these findings are also reinforced and confirmed by performing the performance evaluation of the trained and validated models on a new set of real test data not previously used. This evaluation aims to explore how the models would perform with unseen (future) data from an elite professional soccer season. To fully grasp this part of the discussion, we recommend referring to

The top-performing models included: KNN for Dataset 1 (baseline), CART for Dataset 2, RF for Dataset 3, SVM for Dataset 4, and another SVM for Dataset 5. Arranging them by descending performance, the results align in both testing and the training/validation phases: Dataset 5 stands out again, significantly enhancing decision-making compared to relying solely on the calculation of ratios ‘ ACWR U N C O U P L E D ACWR C O U P L E D ACWR E W M A

The combination of SVM with Dataset 5, therefore, emerges as the top-performing model during the testing phase on ‘ unseen data ’, displaying the following metrics: ROC-AUC = 0.7707 [95% CI: 0.7407–0.8094], G-Mean = 0.7701 [95% CI: 0.7412–0.7986], Accuracy = 0.7434 [95% CI: 0.7095–0.7758], PR-AUC = 0.4509 [95% CI: 0.4168–0.4864], Type I Error = 0.1999 [95% CI: 0.1792–0.2205], and Type II Error = 0.2585 [95% CI: 0.2139–0.2964]. The reported metric values correspond to the best-performing model on the test set, selected according to the predefined evaluation criteria—maximizing ROC-AUC, G-Mean, and PR-AUC, while minimizing Type I and Type II errors—as shown in X testing

To properly interpret the PR-AUC results in an unbalanced dataset, it is essential to recalculate the random classifier baseline using the method described in the literature, which is detailed in X testing

Conversely, classifiers whose PR curves fall more steeply illustrate a rapid decrease in accuracy with increasing recall, indicating less robust performance. Despite this, the fact that the curves do not steeply drop to the baseline suggests that SVM classifiers, in general, exhibit decent performance, albeit with some variability. Among these, some classifiers achieve a good balance between accuracy and recall, making them suitable for broader application across various threshold scenarios. In contrast, others may be more finely tuned for specific conditions, which could limit their effectiveness in different contexts. These differences are captured in the width of the confidence intervals, reinforcing the need for cautious model selection based on the application scenario.

Our best classifier with SVM obtained a PRC-AUC of 0.4509 [95% CI: 0.4168–0.4864], which is significantly higher than the baseline of 0.0329. This suggests that our classifier outperforms a random model by a considerable margin. If the baseline were near 0.5, as in a balanced dataset like the one we created for the training and validation phase (with ROC and PR curves depicted in

With the aim of complement the evaluation based on the best single model performance, we also analyzed the statistical distribution of testing metrics across 20 independently trained models for each algorithm. Each of these models was evaluated on the same hold-out test set composed of unseen data. By computing the mean and the 95% confidence intervals (IC95%) for key evaluation metrics (ROC-AUC, G-Mean, Accuracy, PR-AUC, Type I Error, and Type II Error), we aimed to assess not only peak performance but also the consistency and reliability of each modeling approach. The resulting distributions allowed us to compute mean values and 95% confidence intervals (IC95%) using the t-Student method, providing robust estimates of generalization performance variability.

The analysis, summarized in

In addition to confirming the same conclusions from the training and validation process, results of testing phase proves that our feature engineering approach—the Calculation of Cumulative and Temporal Variations of an External Workload Footprint of a Soccer Player—representing in Dataset 5, is the best proposal in predicting non-contact muscle injuries in real competition scenarios, enhancing decision-making compared to relying solely on the calculation of ratios ‘ ACWR U N C O U P L E D

The papers by Rossi et al. and Carey et al. [

Carey et al. [

Rossi et al. [

In contrast, our work proposes a more rigorous and generalizable framework through the introduction of the Footballer Workload Footprint (FWF), which models training loads as discrete time series matrices and systematically computes cumulative and differential variations. Moreover, 160 models were trained across eight supervised multivariate techniques, ensuring a robust exploration of the predictive space. The application of advanced undersampling, oversampling (SMOTE), dimensionality reduction (PCA), repeated stratified cross-validation, permutation testing, and bias-variance tradeoff analysis provides a stronger empirical foundation for our results.

Our models, particularly those trained with FWF cumulative and temporary variations (Dataset 5), achieved a best ROC-AUC of 0.7707 [95% CI: 0.7407–0.8094] and a best PR-AUC of 0.4509 [95% CI: 0.4168–0.4864] when tested on truly unseen data from the most competitive phase of the season. Additionally, all reported performance metrics were accompanied by 95% confidence intervals, further reinforcing the robustness and statistical validity of our findings.

In terms of model interpretability, Carey et al. provided no explicit interpretative tools, whereas Rossi et al. emphasized the extraction of human-readable rules from decision trees. Our study advances this dimension by integrating visual heatmaps and structured variation matrices (FWF) that facilitate interpretability at the individual player level, aligning predictive outputs with intuitive representations usable by medical and technical staff.

Therefore, the FWF approach not only improves predictive accuracy but also enhances the explainability, robustness, and real-world applicability of injury forecasting models in elite soccer. This methodological advancement suggests a pathway toward the establishment of standardized, reproducible, and scalable frameworks for injury prediction across professional sports contexts.

Given the critical importance of player health and the growing competitive demands on soccer players throughout the season, it is crucial for official bodies such as FIFA and/or UEFA to consider allocating resources towards continued research in this area. Establishing a standardized modeling approach, as FWF matrix proposed in this paper, could facilitate the creation of an global-shared, anonymized and accessible workloads footprints injury database. Such a resource would enable medical teams, physical trainers, and sports data scientists to contribute data and access a centralized repository, thereby enhancing collective efforts in injury prevention and management.

Results are encouraging, as they underscores the validity and potential of our methodology within the context of advanced analytics in sports science, offering a robust alternative that could be implemented on elite soccer teams in the medium term, specially if data shared-banks are consolidated. Our approach not only improves injury prediction ability comparatively to the various ACWR calculations used by experts, but also could paves the way for a more detailed and refined analysis of risk factors, thus optimizing strategies for prevention and intelligent management of workloads for elite soccer players thanks to the FWF’s explanatory capabilities, as depicted in


## 5 Conclusions and future work

In this work, we propose new ideas to model and control the training workload of players, aimed to forecast more effectively non-contact injuries in elite soccer teams, the day before a match. Acute:chronic workload ratio (ACWR) measurement is an accepted practice in elite soccer teams to prevent non-contact injuries and minimize risk, but as we have shown in our paper, it has important shortcomings to track monotony and temporal variations in load. Further, from a machine learning perspective, the inherently imbalanced nature of most collected datasets necessitates additional effort to properly handle the skewed class distribution.

To deal with all of that, we present a new approach to control training load inspired by bilinear modeling [ FWF ( V × E n ) ’ matrix, based on integral and differential calculus. This calculation can be considered the new calculation of ACWR in the era of machine learning era and, for this reason, we also proposed a look at explainability. f k

In order to evaluate our proposal, we designed an exploratory and confirmatory analysis which compared models trained with our new method, versus the models trained using different calculations of ACWR. We carried out it with the standard and most representative techniques of supervised machine learning, facing a real highly imbalanced and complete season dataset from a Spanish First Division (LaLiga) soccer elite team, that also compited on Copa del Rey and European UEFA competitions.

The conducted experiments supported significant improvements and insights. The results are coherent with existing literature, demonstrating that the predictive power of the different training load calculations, such as ‘ ACWR C O U P L E D ACWR U N C O U P L E D ACWR E W M A footprint ’-based models ∑ T F W F Δ T F W F ∑ T F W F Δ T F W F p − v a l u e ≤ 0.001

The main weakness of the results obtained lies in the low PR-AUC, so the models have difficulties to correctly identify the instances of the positive class (the injured), especially in unbalanced datasets, as is the case of the real-world environment of our problem. However, neither most cited papers present an analysis of the Precision Recall curve or the PR-AUC metric, which could further enrich the evaluation of their predictive models in handling class imbalance. We encourage researchers in this field to publish their PR curves and PR-AUC metrics from now on, ideally accompanied by confidence intervals. Doing so would allow not only comparisons of central performance values, but also assessments of stability and statistical reliability, which are essential in high-stakes applications like injury prediction.

The main issue is not the lack of injury samples, as these will naturally become more abundant over time, due to elite clubs allocating more resources in this direction. Let us recall that the objective of physical trainers and medical teams is precisely to avoid the events of the positive class, i.e., that soccer players get injured. Rather, the key difficulty lies in the lack of an established industry standard for gathering, modeling, and processing players workloads and injury data for appropriate reuse in research and/or for creating global-shared, anonymized and public models that can be retrained and refined over time. Our paper illustrates that our proposed method, the Footballer Workload Footprint (FWF), is efficient, practical, and resilient in tackling these challenges.

Results are therefore encouraging, as they underscores the validity and potential of our methodology within the context of advanced analytics in sports science, offering a robust alternative that could be implemented on elite soccer teams in the medium term, specially if shared-data banks are consolidated. However, some limitations must be noted. The present study used data from a single elite Spanish First Division team (LaLiga) over one season, excluding goalkeepers, which may affect the generalizability of the results to other leagues, competitive contexts, or playing styles. Furthermore, variations in GPS technology, differences in internal workload monitoring practices, and non-standardized injury classification across clubs can introduce additional variability that may impact model transferability. Future studies should seek to validate the Footballer Workload Footprint (FWF) approach across multiple teams, seasons, and leagues, ideally using harmonized data collection protocols.

Building on the results obtained and addressing the limitations identified, we propose the following specific lines of future research:

These strategies not only have the potential to improve current metrics but could also contribute new insights into injury prevention within the elite sports context. Furthermore, they may enhance the generalization and practical applicability of injury prediction models, helping to bridge the gap between scientific research and daily decision-making by sports professionals. Also will contribute to the development of more reliable and personalized injury prediction and prevention systems. But the implementation of these proposals will require ongoing interdisciplinary collaboration and expanded access to high-quality datasets, such as the one that was available to us in the present study.


## Supporting information


## Associated Data
