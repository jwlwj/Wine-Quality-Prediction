# Wine-Quality-Prediction
## Introduction
This is the project of my year 3 (2015) Machine Learning coursework in Imperial College London, and was awarded an 'A+' grade. The dataset is obtained from [UCI Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality). 

This project utilised Python to program an algorithm to predict wine quality by a scoring system which based on 10 different input variables. 

Linear regression predictor is generated as a baseline of predictor, and three advanced regression algorithms, which are k-Nearest Neighbours Algorithm, Gradient Boosting for Regression and Support Vector Machine for Regression are applid, under a computationally efficient procedure that performs simultaneous variable and model selection.

## Results:
### White Wines:
\begin{table}[]
    \centering
    \begin{tabular}{c|c|c}
        Algorithms & Accuracy & RMSE  \\
        \hline
        KNN & 85.2% & 0.721 \\
        GBR & 87.5% & 0.676 \\
        SVM & 85.2 & 0.743 \\
    \end{tabular}
\end{table}
