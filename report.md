### Overview
This analysis investigated EEG frequency patterns distinguishing Alzheimer's Disease (AD) patients from age-matched healthy controls, with attention to methodological approaches and their implications.
### Methodology and Results
#### Feature Selection Process
After analyzing all available frequency bands and regions, I employed mutual information (MI) analysis to systematically identify the most discriminative features. Starting with over 50 potential frequency-region combinations, I selected the features that provided the strongest and most complementary diagnostic signals. MI scores range from 0 to 1, where 0 indicates no mutual information (independence) and 1 indicates perfect mutual information (dependence).
##### Selected Features and Their Importance
| Feature Band       | Frequency Range | Region   | MI Score |
| ------------------ | --------------- | -------- | -------- |
| Theta (slow range) | 3-5 Hz          | Global   | 0.112    |
| Delta              | 1-3 Hz          | Temporal | 0.104    |
| Alpha (fast range) | 10-13 Hz        | Central  | 0.099    |
| Beta (slow range)  | 13-20 Hz        | Parietal | 0.082    |
| Beta (fast range)  | 21-30 Hz        | Global   | 0.037    |

The MI scores obtained reveal a clear hierarchical pattern of frequency band importance in AD diagnosis. The slow-wave frequencies show the strongest associations, with Theta (MI = 0.112) and Delta (MI = 0.104) bands together accounting for over 56% of the relative importance. These findings align with previous literature showing increased slow-wave activity in AD. The analysis also reveals region-specific importance, particularly in temporal and central areas for Delta and Alpha bands respectively.

The relatively close MI scores among the top features (0.082-0.112) suggest that AD detection benefits from considering multiple frequency bands rather than relying on a single discriminative feature.


MI was chosen for its ability to:
- Capture non-linear relationships between frequency patterns and AD diagnosis
- Identify subtle alterations in frequency distributions characteristic of AD
- Provide interpretable importance measures that are robust to scaling and transformations
- Quantify both linear and non-linear dependencies without assuming a particular distribution
#### Model Development
##### Model Selection
Using Optuna, I evaluated multiple classifier types:
- XGBoost
- CatBoost
- Gradient Boosting
- Random Forest
- Support Vector Machines

CatBoost emerged as the best performing model.
##### Winning CatBoost Model Configuration

| Parameter              | Value   |
| ---------------------- | ------- |
| Iterations             | 415     |
| Learning Rate          | 1.76e-4 |
| Depth                  | 10      |
| L2 Leaf Regularization | 4.75e-8 |

I explored two approaches, each with different implications for clinical application:
1. Conservative Approach (Downsampling)
    - Sensitivity: 77% (ability to correctly identify AD patients)
    - Specificity: 81% (ability to correctly identify healthy controls)
2. Oversampling Approach
    - Sensitivity: 82% (ability to correctly identify AD patients)
    - Specificity: 85% (ability to correctly identify healthy controls)
###  Insights
#### The Trade-off of Detection vs. Reliability
My exploration revealed a fundamental trade-off in detection approaches that has potential clinical implications:

1. High-Detection Priority (Oversampling)
    - Achieves higher AD detection rate (82% sensitivity). Done by oversampling AD cases to match larger amount of healthy patients in the dataset.
    - This prioritizes catching potential AD cases, but at the potential cost of more false positives.
    - Best suited for screening contexts where follow-up testing is readily available
2. Reliability Priority (Conservative Approach)
    - Shows slightly lower detection rate (77% sensitivity) but bases all training and predictions on real AD patient data
    - Prioritizes prediction confidence over catch-all detection, potentially missing some early cases but providing more reliable positive predictions
    - Better suited for contexts where false positives are costly or could cause stress.

While oversampling achieved the highest performance metrics, I recommend the more conservative downsampling approach for clinical implementation. This choice prioritizes reliability and generalizability over raw performance metrics. 

 
