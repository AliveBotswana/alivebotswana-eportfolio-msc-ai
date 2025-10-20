# alivebotswana-eportfolio-msc-ai
Machine Learning Module - E-Portfolio
University of Essex - MSc in Artificial Intelligence
Student: Mosweu Boitshepo
Module: Machine Learning

📋 Overview
This repository contains all artefacts and coursework completed during the Machine Learning module. The work demonstrates progression from foundational exploratory data analysis to advanced deep learning applications, covering supervised learning, unsupervised learning, and ethical considerations in ML deployment.

📂 Repository Structure
Unit 2 - Exploratory Data Analysis
Assignment_for_Auto_MPG.ipynb
- Exploratory data analysis on the Auto-mpg dataset
- Identifies missing values, calculates skewness (0.457) and kurtosis (-0.510)
- Generates correlation heatmaps (mpg vs. weight: -0.83)
- Creates scatter plots and removes outliers
- Encoding categorical variables (origin)



Unit 3 & 4 - Correlation and Regression
Correlation_&_Regression.ipynb
- Pearson correlation analysis (coefficient: 0.888)
- Linear regression predictions
- Multiple linear regression for CO2 emissions
- Calculates coefficients for weight and volume variables
- Jaccard similarity coefficients for pathological tests



Unit 6 - Team Project (Classical ML)
Group Work
EDA_Group_Assignment.ipynb
- Comprehensive EDA on AB_NYC_2019.csv
- Price distribution analysis (right-skewed, outliers to $10,000)
- Neighborhood pattern analysis (Manhattan highest prices)
- Occupancy rate visualization (bimodal distribution)

Clustering_Group_Assignment.ipynb
- K-Means clustering implementation (k=5)
- Elbow Method and Silhouette Score analysis
- Market segmentation by neighborhood, room type, and occupancy
- Integration with regression models (GBM achieved MAE $44)

Unit 11 - Individual Assignment
Track 1: Classical Machine Learning
Track_1_Individual_Assignment.ipynb
- Support Vector Machine (SVM) implementation - 57.1% accuracy
- K-Nearest Neighbors (KNN) implementation - 53.7% accuracy
- Comparison with deep learning approaches
- Class-wise performance analysis

Track 2: Deep Learning
Track_2_Individual_Assignment.ipynb
- Custom CNN architecture for CIFAR-10 object recognition
- Four convolutional blocks with batch normalization and dropout
- Achieved 88% validation accuracy at epoch 82
- Transfer learning with ResNet50 (45% accuracy)
- Hyperparameter tuning via random search
- Data augmentation techniques (rotations, flips, brightness adjustments)
- Confusion matrix and F1-score analysis

Unit 12 - Extended Learning
Elephant_DetectionDraft.ipynb
Aerial elephant census detection using YOLOv8
- Custom synthetic dataset for wildlife monitoring
- Performance metrics: mAP50 (0.167), precision (0.416), recall (0.183)
- 63 epochs with early stopping at epoch 43
- Demonstrates real-world deployment challenges




🎯 Learning Outcomes Demonstrated

Articulate legal, social, ethical and professional issues faced by ML professionals
- Ethical analysis of Airbnb pricing algorithms and housing inequality
- Surveillance ethics in wildlife monitoring
- GDPR compliance and data governance considerations

Understand applicability and challenges of different datasets
- Handling skewed distributions (Airbnb data)
- Low-resolution image limitations (CIFAR-10)
- Domain gap challenges (synthetic to real-world data)

Apply and critically appraise ML techniques to real-world problems
- Comparative analysis: Deep learning (88%) vs. Classical ML (57.1%)
- Critical evaluation of model limitations and interpretability
- Performance-interpretability trade-offs

Effective team membership in virtual professional environments
- Collaborative development with version control
- Cross-time zone coordination
- Peer feedback integration

🛠️ Technologies & Libraries Used

Languages: Python
ML/DL Frameworks: TensorFlow/Keras, Scikit-learn, YOLOv8
Data Processing: Pandas, NumPy, Papaparse
Visualization: Matplotlib, Seaborn
Statistical Analysis: SciPy, Statsmodels
Development Tools: Jupyter Notebook, Git/GitHub
