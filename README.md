# From Trees to Forests: Exploring the Power of Random Forest in Machine Learning
Hello, machine learning enthusiasts! Today, I'm excited to share the second installment in my series on random forests. Before delving into this, I highly recommend reading [my first article]( https://kaychansiri.github.io/ClassificationTree/), which focuses on decision trees, particularly classification trees. This foundational knowledge is crucial before attempting to build your first random forest (RF) algorithm. For those well-versed in decision trees, let's explore RF today.

## What is RF?
RF is a robust machine learning algorithm designed to mitigate issues commonly associated with decision trees, such as overfitting and the imbalance between bias and variance. One of the reasons RF is among my favorite algorithms is its minimal preprocessing requirements. The scale of features is largely irrelevant because the algorithm processes each feature orthogonally, seeking optimal split points. For instance, if one feature measures height in centimeters and another weight in pounds, RF can still efficiently determine the best split for each. This differs from algorithms like K-Nearest Neighbors (KNN), where preprocessing or standardizing feature scales is essential to avoid skewed groupings due to disproportionate feature scales, potentially leading to poor model performance or interpretational errors. Although RF generally requires more computational time than KNN, it excels with features of varying scales and high-dimensional datasets, avoiding the pitfalls of dimensionality that KNN faces. Please refer to [my post](https://www.linkedin.com/posts/kay-chansiri_one-common-mistake-in-applying-k-nearest-activity-7158197126349340672-lZ3p?utm_source=share&utm_medium=member_desktop) on the curse of dimensionality for more details.

## Types of RF
Before elaborating on the concept of RF, let's discuss the types of RF available:

1. **Random Forest Classifier**
In classification tasks, each decision tree aims to classify samples into categories and reduce node impurity using measures like the Gini index or entropy. These metrics help the algorithm determine how to best split the data at each node to increase the homogeneity of the resulting child nodes relative to the target class. By minimizing impurity or maximizing information gain at each split, the cost of misclassification within each tree is effectively reduced.

2. **Random Forest Regressor**
In regression tasks, the objective is to predict a continuous value and minimize the variance within each node after the split. Here, impurity is represented by the variance in the node's values. A preferred split is one that results in child nodes with lower variance than the parent node, thus grouping together similar scores.

## Bagging 
Fundamentally, RF combines decision trees with a bootstrapping method. Simply put, each tree in your forest is built using bootstrapping to generate a new sample from your entire dataset.This new sample maintains the same probability distribution as the original dataset. For example, if you are analyzing a Netflix dataset with high variance in viewer preferences, your subsamples will mirror this distribution.

> A common misconception about RF is that each tree utilizes a smaller dataset than the original. In reality, bootstrapping employs a sampling-with-replacement technique, ensuring each new dataset also contains, says, 100,000 cases if the original did. It is possible for a single case to appear multiple times in a subsample since each resampling could select and replace it repeatedly.

<img width="752" alt="Screen Shot 2024-05-01 at 7 48 35 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/916f6260-7e59-4f01-90d2-80c745b0c77c">


Bootstrapping generates subsamples that replicate the original data distribution, allowing for numerous samples without the need for additional data collection. Each subsample is utilized to construct a tree. When these trees are combined, or ensembled, they produce a more accurate average prediction for regression tasks or the most frequently predicted class in classification tasks. This ensemble approach helps mitigate the high variance that single decision trees might exhibit. This whole process is referred to as **'bagging'** or **'boostraping aggregation.'**

It's important to recognize that errors in ML algorithms can stem from three sources: variance (indicative of overfitting), bias (indicative of underfitting), and noise (unpredicted variance of the target), as shown in the equation below

<img width="695" alt="Screen Shot 2024-05-01 at 7 32 10 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/7439323c-a8cd-4496-b653-15ca800506ff">


The goal of bagging is to reduce the variance term to make *h*<sub>*d*</sub>(X)(predicted values) as close as possible to *h*(X) (observed values). As shown in Figure 1, with an aim to reduce variance without increasing bias, bagging, or Bootstrap Aggregating, involves sampling m datasets with replacement from the initial data pool, *D*. This process generates datasets  *D*<sub>*1*</sub>,  *D*<sub>*2*</sub>,..., *D*<sub>*m*</sub>. For each, *D*<sub>*i*</sub>, train a classifier or a regressor *h*<sub>*i*</sub>(). The final classifer or regressor is calculated as: 

<img width="231" alt="Screen Shot 2024-05-01 at 8 03 10 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/77885497-5500-4b90-a1d6-8fa2218973e4">








