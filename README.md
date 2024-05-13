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

Note that a larger *m* results in a better ensemble and prediction accuracy. However, if *m* is too large, you may eventually end up modeling noise and slowing down the computation.

## Feature Selection

In addition to the bootstrapping process, which involves randomly selecting samples (i.e., rows in the dataset) to reduce variance, Random Forests (RF) also randomly select features at each node of each tree in the forest to further reduce variance without increasing bias. This raises a key question: How many features should be used for each split in an RF?

The number of features to consider at each split in an RF is a tunable parameter known as **`max_features`**. A common rule of thumb is to use the square root of the total number of features for classification tasks, and about one-third of the total features for regression tasks. For example, if your dataset comprises 25 features, you might set approximately 5 features (the square root of 25) for a classification task or about 8 features (one-third of 25) for a regression task. 

During the construction of each tree, at each node, a subset of features is randomly selected based on the `max_features` parameter. The best split on these features is then determined based on how well it can separate the classes in classification or reduce variance in regression. Once a tree begins to form, its structure proceeds to branch out based on the best splits at each node and does not backtrack to alter previous decisions.

Each tree in the forest might end up using different features at its root, contributing to the diversity of the models within the forest. It's important to note that the random selection of features in RF distinguishes it from the approach in a single decision tree, where potentially all features could be used at any node if the tree depth is not limited.

Two major parameters that you can fine-tune in a Random Forest are `max_features` and the number of trees **(`n_estimators`)**. Additionally, modern computational capabilities allow each tree to be built in parallel, which significantly saves on computational time without affecting the statistical performance of the model. This aspect of Random Forests is particularly beneficial for feature selection, as the algorithm can effectively identify which predictors are most impactful by evaluating the reduction of impurities across the forest.

## Out-of-bag error estimation

When building each tree in RF, the algorithm randomly selects a subset of the data (with replacement) for training, which means some data points may be selected multiple times, while others may not be selected at all. The data points not used to train a particular tree are known as the **"out-of-bag"** (OOB) data for that tree. To further explain, with the bootstrapping method, each sub-dataset automatically excludes some data points. For example, from bootstrap 1, the first few data points could be participant IDs 1, 2, 3, and 4, and then for bootstrap 2, it could be IDs 1, 2, 2, and 4, as we sampled with replacement. In this case, participant 3 got automatically excluded from bootstrap 2 while participant ID 2 got selected repeatedly. Approximately, each sub-dataset has 60% overlap of the samples with the original dataset, and the other 40% are repeated cases. 

The main advantage of OOB error estimation is that it provides a way to estimate the model's performance without needing a separate validation or test set. In other words, you may not have to run the cross-validation technique, which could take a lot of time if you have a big forest, to evaluate the model's performance. Refer to my [previous post](( https://kaychansiri.github.io/ClassificationTree/) to learn more about the cross-validation (CV) process. To provide a brief summary, CV is the process that helps evaluate the model's performance, not to reduce variance like the bagging process. As As both OOB error estimation and CV involve categorizing the data pool into subsets, people often get confused between these two methods.

<img width="759" alt="Screen Shot 2024-05-02 at 2 24 56 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/22e97df1-dcbc-42ec-afe2-05958b7123ac">


The OOB error is very useful as it is automatically available as a by-product of the training process of the random forest, requiring no additional computational cost. This makes the OOB error a convenient and efficient tool for model evaluation and tuning, especially when dealing with large datasets where cross-validation can be computationally expensive. 
> Note that beyond bagging, other methods such as collecting more data points or applying regularization techniques can help reduce bias as well. Another thing to note is that bagging is not exclusive to random forests; this ensemble technique can be applied to other algorithms as well if your goal is to reduce variance.

## Features to be Fine-Tuned
I previously mentioned the number of trees (`n_estimators`) and the maximum number of features (`max_features`) as parameters that can be fine-tuned. Below is the list of other parameters. For all parameters, the 'GridSearchCV' function, utilizing cross-validation, could help identify the best values along with other factors, such as computational complexity, time, and project objectives. For data with high dimensions, using GridSearch may not be time- and computing-efficient. Other strategies, such as random search or Bayesian optimization, are recommended.
* **Maximum Depth of Trees (`max_depth`)**: There is no exact rule of thumb regarding how deep a tree should be, although limiting trees to not be too deep helps prevent overfitting. However, setting the parameter too low could lead to underfitting.
* **Minimum Samples Split (`min_samples_split`)**: Typically, higher values prevent the model from learning overly specific patterns, which can lower the risk of overfitting. However, a value that is too high could lead to underfitting.
* **Minimum Samples Leaf (`min_samples_leaf`)**: The minimum number of samples required to be at a leaf node. Setting a too low number can lead to overfitting. However, setting the number too high could lead to underfitting. Justifying the right number also depends on your total sample size and whether you have a balanced or imbalanced design, especially for a classification forest. If you have an imbalanced design (i.e., the number of one target class is way higher than the other for a binary classification forest), considering the class with the smaller sample size to decide the minimum sample of the tree could help prevent underfitting issues from the tree not being able to grow due to a smaller sample size of the class compared to the predefined minimum sample split.
* **Class Weight (`class_weight`)**: Relevant to the point above, setting class weight is useful if you are building a classification forest with imbalanced classes of the target output. The parameter associates classes with weights and prioritizes the minority class during training to compensate for an imbalanced dataset. There are several other methods to deal with the class imbalance issue, which I will discuss more in my upcoming post.
* **Bootstrap (`bootstrap`)**: By default, RF always utilizes bootstrap. However, if the function is set to be 'false', the whole dataset is used to build each tree. I do not recommend this as bootstrapping can generate heterogeneous samples that still share the same distribution with the original data pool to prevent the issue of overfitting commonly found in decision trees where the whole dataset is used.
* **Criterion (`criterion`)**: The function to measure the quality of a split. For classification forests, 'Gini' for Gini impurity and 'entropy' for information gain are commonly used. For regression forests, Mean Squared Error ('MSE') or Mean Absolute Error ('MAE') are used to calculate the distance between the actual and predicted values at a node. The split that minimizes this error is chosen.
* **Max Leaf Nodes (`max_leaf_nodes`)**: The maximum number of leaf nodes a tree can have. Note that if this parameter is defined, the trees might not reach the `max_depth` specified as the algorithm considers growing each tree according to its max_leaf_node first.
* **Random State (`random_state`)**: Finally, DO NOT forget to set your random state (or set.seed() if you are an R user). This is important for model reproducibility!

## Now that you have learned about the basics of RF, let's apply the algorithm to a real-world use case
In today exmaple, we'll explore how a nationwide satellite service provider uses customer feedback and operational data to enhance customer experience and optimize service delivery. The company, which operates across various counties in California including Los Angeles, San Francisco, and Orange County, seeks to understand the dynamics that influence customer satisfaction and thus improve their services accordingly.

### Dataset Overview
The dataset consists of responses and operational metrics collected from customers who have interacted with the service provider. Here are some key features of the dataset:
* customer_id: Unique identifier for each customer.
* feedback_phase: Indicates the phase of feedback collection (phase 1 and 2), aligning with different stages of the customer journey.
* race_caucasian, race_african_american, race_other, ethnicity_hispanic: Demographic information of the customer to monitor diversity and inclusiveness in service impact.
* customer_female: Binary indicator of the customer's gender, coding female customers as 1 to ensure gender-specific service considerations.
* age: Customer's age to tailor services according to different age groups.
* customer_response, manager_response, representative_response: Feedback scores from the customer, the service manager, and the customer representative, respectively, which reflect different perspectives on the service delivered.
* CEO_oversee: Indicates whether the service case was directly overseen by the CEO, used as a measure of high-priority service handling.
* issue_count_during_service: Counts of any issues reported during the service provision, indicating the complexity or challenges faced.
* years_experience_representative: Experience level of the representative assigned to the customer, hypothesizing that more experienced representatives deliver better service.
* satisfaction_rating: Overall customer satisfaction rating, serving as the outcome variable for our analysis.
* county_LA, county_SF, county_OC: Binary indicators representing the location of service.
* service_location_customer_home, service_location_business_address, service_location_community_spaces, service_location_apartments: Categorial data indicating where the service was provided, which could impact customer satisfaction.

### Data Preparation

The first step is you need to ensure that each feature in the dataset is encoded in the way they are supposed to be (i.e., categorical features are encoded as categorical and continuous are encoded as continuous).

```ruby
data.dtypes
```
<img width="409" alt="Screen Shot 2024-05-12 at 7 23 35 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/32272b6e-a7f2-4a7c-8aca-43d7e9357115">


You can see that certain categorical variables are are still coded as numeric. Let's convert them: 

```ruby
#Convert int64 to be categorical 
# List of categorical variables
categorical_variables = [
    'feedback_phase', 'race_caucasian', 'race_african_american', 'race_other',
    'ethnicity_hispanic', 'customer_female', 'customer_response', 'manager_response',
    'representative_response', 'CEO_oversee', 'county_LA', 'county_SF', 'county_OC',
    'service_location_customer_home', 'service_location_business_address',
    'service_location_community_spaces', 'service_location_apartments'
]

# Converting columns to 'category' dtype and assigning back to the DataFrame
data[categorical_variables] = data[categorical_variables].astype('category')
```
Check the type of each feature again to ensure the conversion works. 

```ruby
data.dtypes
```
<img width="436" alt="Screen Shot 2024-05-12 at 7 26 54 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/bce74fbc-7bdd-441f-86ca-85cd8eef3131">

The next step is to check missing values. 

```ruby
# Calculate the percentage of missing values in each column
missing_percentage = (data.isna().sum() / len(data)) * 100

# Print the percentage of missing values per column
print(missing_percentage)
```
<img width="384" alt="Screen Shot 2024-05-12 at 7 37 16 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/30cd4241-b7fd-417c-81b4-90e18b3dba12">

*****Correct grammar starting from here ***
The output reflects zero misssing percentages becasue I have dealt with the missingness prior to doing this demo. If you data has missing values and the percentage for each missing column is small (< 2%) you may consider a simple imputation strategy such as mean or mode imputation for continuous and categorical features, respectivelly. If the missingness is large, you have to find out if the missingness is at random (i.e.,not related to other features in the model) or non-random (i.e., significantly related to other features in the model). If it's the latter case, a more complex imputation technique such as multiple imputation, expecatiom maximization, or K-nearest neirghbor could be utilized to ensure that the imputed values do not alter your data pattern. 

Now that we prepared the data, let's split the data into a training and testing set. If you still rememeber from what I described before, the data is longitudial. Thus, we will use the samples from the first wave of customer satifsaction measurement as the training data and use the second wave as the testing data. This method enables us to exmaine whether the model built on a previous time point can be used to predict future data points. Although this method can deal with the temporal nature of the dataset, there are flaws as the method does not account for within-subject effects (i.e., the change of satisfaction score within invividuals over time). The method also considers each observation in the dataset as the unit of analysis, ignoring the analysis at the subject level where time is clustered in. However, for simplicity in doing the demo, I will carry on with trying to deal with the time factor as best as I can by randomly dividing the data into the traning and tetsing set. Then for each ID in the traning set, selecting their rows where 'feedback_phase' == 1 to be used. For the testing set, selecting their rows where 'feedback_phase' == 2 to be used.

```ruby
#Separate traning and testing data
from sklearn.model_selection import train_test_split

# Get unique IDs
unique_ids = data['customer_id'].unique()

# Split IDs into training and testing groups
train_ids, test_ids = train_test_split(unique_ids, test_size=0.4, random_state=42)

# Select the data for training and testing
train_data = data[(data['customer_id'].isin(train_ids)) & (data['feedback_phase'] == 1)]
test_data = data[(data['customer_id'].isin(test_ids)) & (data['feedback_phase'] == 2)]
```

There are better methods to deal with longitudinal clustered data such as mixed level modeling (referred to (this post I wrote)[https://github.com/KayChansiri/Demo_Longtitudinal-Multilevel-Modeling] or random forest extensions. Hu and Szymczak wrote a very good (article)[https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10025446/]. I highly recommend you to read through their article to better understand limitations of and strategies when applying RF with longitudinal data. 

Now back to our  business, after splitting the data into the traning and testing set, let's use gridsearch to find the best values for `max_depth`,`min_samples_split`, `min_samples_leaf`, `criterion`, and `max_leaf_nodes`. Note that some of the parameters I mentioed previously (e.g., `class_weight`) are not fine tuned here as we are working with a regression tree.





