# Wine-Quality-Prediction
## Introduction
This is the project of my year 3 (2015) Machine Learning coursework in Imperial College London, and was awarded an 'A+' grade. The dataset is obtained from [UCI Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality). 

This project utilised Python to program an algorithm to predict wine quality by a scoring system which based on 10 different input variables. 

Linear regression predictor is generated as a baseline of predictor, and three advanced regression algorithms, which are k-Nearest Neighbours Algorithm, Gradient Boosting for Regression and Support Vector Machine for Regression are applid, under a computationally efficient procedure that performs simultaneous variable and model selection.The Gradient Boosting for Regression achieved best re- sults, outperforming the remaining algorithms.

## Results:
### Pairplot For Red WInes and White Wines
![pairplot
### White Wines:
| Algorithms | Accuracy | RMSE |
| :---:  | :---:  | :---:  |
|KNN | 85.2% | 0.721 |
|GBR | 87.5% | 0.676 |
|SVM | 85.2% | 0.743 |
### Red Wines:
| Algorithms | Accuracy | RMSE |
| :---:  | :---:  | :---:  |
|KNN | 88.0% | 0.677 |
|GBR | 90.9% | 0.614 |
|SVM | 88.4% | 0.742 |
### Combined Dataset
| Algorithms | Accuracy | RMSE |
| :---:  | :---:  | :---:  |
|KNN | 86.2% | 0.709 |
|GBR | 88.7% | 0.651 |
|SVM | 85.7% | 0.724 |
