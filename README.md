# From Trees to Forests: Exploring the Power of Random Forest in Machine Learning
Hello, machine learning enthusiasts! Today, I'm excited to share the second installment in my series on random forests. Before delving into this, I highly recommend reading [my first article](https://github.com/KayChansiri/Demo_Classification_Tree), which focuses on decision trees, particularly classification trees. This foundational knowledge is crucial before attempting to build your first random forest (RF) algorithm. For those well-versed in decision trees, let's explore RF today.

## What is Random Forest?
RF is a robust machine learning algorithm designed to mitigate issues commonly associated with decision trees, such as overfitting and the imbalance between bias and variance. One of the reasons RF is among my favorite algorithms is its minimal preprocessing requirements. The scale of features is largely irrelevant because the algorithm processes each feature orthogonally, seeking optimal split points. For instance, if one feature measures height in centimeters and another weight in pounds, RF can still efficiently determine the best split for each. This differs from algorithms like K-Nearest Neighbors (KNN), where preprocessing or standardizing feature scales is essential to avoid skewed groupings due to disproportionate feature scales, potentially leading to poor model performance or interpretational errors. Although RF generally requires more computational time than KNN, it excels with features of varying scales and high-dimensional datasets, avoiding the pitfalls of dimensionality that KNN faces. Please refer to [my post](https://www.linkedin.com/posts/kay-chansiri_one-common-mistake-in-applying-k-nearest-activity-7158197126349340672-lZ3p?utm_source=share&utm_medium=member_desktop) on the curse of dimensionality for more details.

## Types of RF
Before elaborating on the concept of RF, let's discuss the types of RF available:

1. **Random Forest Classifier**
In classification tasks, each decision tree aims to classify samples into categories and reduce node impurity using measures like the Gini index or entropy. These metrics help the algorithm determine how to best split the data at each node to increase the homogeneity of the resulting child nodes relative to the target class. By minimizing impurity or maximizing information gain at each split, the cost of misclassification within each tree is effectively reduced.

2. **Random Forest Regressor**
In regression tasks, the objective is to predict a continuous value and minimize the variance within each node after the split. Here, impurity is represented by the variance in the node's values. A preferred split is one that results in child nodes with lower variance than the parent node, thus grouping together similar scores.

## Bagging (Boostrapping Agrreggation) 
Fundamentally, RF combines decision trees with a bootstrapping method. Simply put, each tree in a forest is built using bootstrapping to generate a new sample from the entire original dataset.This new sample maintains the same probability distribution as the original dataset. For example, if you are analyzing a Netflix dataset with high variance in viewer preferences, your subsamples will mirror this distribution.

> A common misconception about RF is that each tree utilizes a smaller dataset than the original. In reality, bootstrapping employs a sampling-with-replacement technique, ensuring each new dataset has a similar sample size to the original dataset. It is possible for a single sample or case to appear multiple times in a subsdataset since each resampling could select and replace cases repeatedly.

<img width="752" alt="Screen Shot 2024-05-01 at 7 48 35 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/916f6260-7e59-4f01-90d2-80c745b0c77c">

Each subsample in the boostrapping process is utilized to construct a tree. When these trees are combined, or ensembled, they produce a more accurate average prediction for regression tasks or the most frequently predicted class in classification tasks than a single tree. In other words, the ensemble process reduce errors better than a decision tree.

It's important to recognize that errors in ML algorithms can stem from three sources: variance (indicative of overfitting), bias (indicative of underfitting), and noise (unpredicted variance of the target), as shown in the equation below

<img width="695" alt="Screen Shot 2024-05-01 at 7 32 10 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/7439323c-a8cd-4496-b653-15ca800506ff">


The goal of bagging is to reduce the variance term to make *h*<sub>*D*</sub>(X)(predicted values) as close as possible to *h*(X) (observed values). With an aim to reduce variance without increasing bias, bagging involves sampling m datasets with replacement from the initial data pool, *D*. This process generates datasets  *D*<sub>*1*</sub>,  *D*<sub>*2*</sub>,..., *D*<sub>*m*</sub>. Each *D*<sub>*i*</sub> then is trained with a classifier or a regressor *h*<sub>*i*</sub>(). The final classifer or regressor across all trees is calculated as: 

<img width="231" alt="Screen Shot 2024-05-01 at 8 03 10 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/77885497-5500-4b90-a1d6-8fa2218973e4">

Note that a larger *m* results in a better ensemble and prediction accuracy. However, if *m* is too large, you may eventually end up modeling noise and slowing down the computation.

## Feature Selection

In addition to the bootstrapping process, which involves randomly selecting samples (i.e., rows in the dataset) to reduce variance, RF also randomly select features at each node of each tree in the forest to further reduce variance without increasing bias. This raises a key question: How many features should be used for each split in an RF?

The number of features to consider at each split in an RF is a tunable parameter known as **`max_features`**. A common rule of thumb is to use the square root of the total number of features for classification tasks, and about one-third of the total features for regression tasks. For example, if your dataset comprises 25 features, you might set approximately 5 features (the square root of 25) for a classification task or about 8 features (one-third of 25) for a regression task. 

During the construction of each tree, at each node, a subset of features is randomly selected based on the `max_features` parameter. The best split on these features is then determined based on how well it can separate the classes in classification or reduce variance in regression. Once a tree begins to form, its structure proceeds to branch out based on the best splits at each node and does not backtrack to alter previous decisions.

Each tree in the forest might end up using different features at its root, contributing to the diversity of the models within the forest. It's important to note that the random selection of features in RF distinguishes it from the approach in a single decision tree, where potentially all features could be used at any node if the tree depth is not limited.

So far, I have introduced parameters that you can fine-tune in RF, including `max_features` and the number of trees (`n_estimators`). Modern computational capabilities allow each tree to be built in parallel, which significantly saves on computational time without affecting the performance of the model. This aspect of RF is particularly beneficial for feature selection, as the algorithm can effectively identify which predictors are most impactful by evaluating the reduction of impurities across the forest.

## Out-of-Bag Error Estimation

When building each tree in RF, the algorithm randomly selects a subset of the data (with replacement) for training, which means some data points may be selected multiple times, while others may not be selected at all. The data points not used to train a particular tree are known as the **"out-of-bag"** (OOB) data for that tree. 

To further explain, with the bootstrapping method, each subdataset automatically excludes some data points. For example, from bootstrap 1, the first few data points could be participant IDs 1, 2, 3, and 4, and then for bootstrap 2, it could be IDs 1, 2, 2, and 4, as we sampled with replacement. In this case, participant 3 got automatically excluded from bootstrap 2 while participant ID 2 got selected repeatedly. Approximately, each sub-dataset has 60% overlap of the samples with the original dataset, and the other 40% are repeated cases. Note that for each predictor (classifier and regressor), the repeated 40% are not the same.

The main advantage of OOB error estimation is that it provides a way to estimate the model's performance without needing a separate validation or test set. In other words, you may not have to run the cross-validation (CV) technique, which could take a lot of time if you have a big forest, to evaluate the model's performance. 

<img width="759" alt="Screen Shot 2024-05-02 at 2 24 56 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/22e97df1-dcbc-42ec-afe2-05958b7123ac">


The OOB error is very useful as it is automatically available as a by-product of the training process of the random forest, requiring no additional computational cost. This makes the OOB error a convenient and efficient tool for model evaluation and tuning, especially when dealing with large datasets where cross-validation can be computationally expensive. 

## Hyperparameters
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

The first step is to ensure that each feature in the dataset is encoded in the way they are supposed to be (i.e., categorical features are encoded as categorical and continuous are encoded as continuous).

```ruby
data.dtypes
```
<img width="409" alt="Screen Shot 2024-05-12 at 7 23 35 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/32272b6e-a7f2-4a7c-8aca-43d7e9357115">


According to the output, certain categorical variables are are still coded as numeric. Let's convert them: 

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

<img width="421" alt="Screen Shot 2024-05-14 at 4 38 14 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/ee397fbe-7b91-4c20-a30c-de0f5d48c848">


The output reflects 0.267% missingness in the years_experience_representative feature. If your data has missing values and the percentage for each missing column is small (<2%), you may consider a simple imputation strategy such as mean or mode imputation for continuous and categorical features, respectively. If the missingness is large, you need to determine whether the missingness is at random (i.e., not related to other features in the model) or non-random (i.e., significantly related to other features in the model). If it is the latter case, a more complex imputation technique such as multiple imputation, expectation maximization, or K-nearest neighbor could be utilized to ensure that the imputed values do not alter your data patterns. For the current exmaple, I use mean imputation:

```ruby
from sklearn.impute import SimpleImputer

# Impute missing values in 'years_experience_representative' with the mean
imputer = SimpleImputer(strategy='mean')
data['years_experience_representative'] = imputer.fit_transform(data[['years_experience_representative']])

# Print the first few rows of the imputed dataset to verify
print(data.head())

```
<img width="393" alt="Screen Shot 2024-05-14 at 4 40 47 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/56061f8a-e3a6-4ed4-85fd-a1de26224eae">


### Machine Learning Operation

Now that we have prepared the data, let's split it into a training and testing set. 

```ruby

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Print the resulting shapes of the training and testing sets
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)


```

<img width="270" alt="Screen Shot 2024-05-14 at 4 42 24 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/d76ab319-ad3e-4abc-8bdc-6392aa189718">


Note that the current dataset is cross-sectional, where each sample is measured their features and outcomes for only time. This might not be the case for a real-world dataset. If you have longitudinal data, there are better methods than RF, such as mixed-level modeling (referred to in [this post](https://github.com/KayChansiri/Demo_Longtitudinal-Multilevel-Modeling) I wrote).  For random forest extensions, Hu and Szymczak wrote a very good [article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10025446/]) regarding how to use the methods to deal with longtudinal, clustered data. I highly recommend reading their article to better understand the limitations of and strategies for applying random forests to longitudinal data.

Now back to our business, after splitting the data into the training and testing sets, let's fit the model using GridSearchCV to find the best values for max_depth, min_samples_split, min_samples_leaf, criterion, and max_leaf_nodes. Note that some of the parameters I mentioned previously (e.g., class_weight) are not fine-tuned here as we are working with a regression forest.

```ruby
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


# Separate features and target variable for training data
X_train = train_data.drop(columns=['satisfaction_rating', 'customer_id', 'feedback_phase']) #not plan to use for the analysis
y_train = train_data['satisfaction_rating']

# Separate features and target variable for testing data
X_test = test_data.drop(columns=['satisfaction_rating', 'customer_id', 'feedback_phase'])
y_test = test_data['satisfaction_rating']

# Define the model
rf = RandomForestRegressor(n_jobs=-1)
```

To speed up the process, I set n_jobs=-1, which basicually instructing the model to use all available CPU cores. Before we move forward with fine-tuning the parameters, let's examine the sample size in each category of our categorical predictors. This is helpful in identifying certain parameters such as min_samples_split.


```ruby
# List of categorical variables
categorical_variables = [
    'feedback_phase', 'race_caucasian', 'race_african_american', 'race_other',
    'ethnicity_hispanic', 'customer_female', 'customer_response', 'manager_response',
    'representative_response', 'CEO_oversee', 'county_LA', 'county_SF', 'county_OC',
    'service_location_customer_home', 'service_location_business_address',
    'service_location_community_spaces', 'service_location_apartments'
]

# Loop through each categorical variable and print the value counts
for variable in categorical_variables:
    print(f"Counts for {variable}:")
    print(data[variable].value_counts())
    print("\n")  # Adds a newline for better readability between outputs
```

Here is the output: 

```
Counts for race_caucasian:
0    8365
1    7364
Name: race_caucasian, dtype: int64


Counts for race_african_american:
0    9567
1    6162
Name: race_african_american, dtype: int64


Counts for race_other:
0    15245
1      484
Name: race_other, dtype: int64


Counts for ethnicity_hispanic:
0    14010
1     1719
Name: ethnicity_hispanic, dtype: int64


Counts for customer_female:
0    7965
1    7764
Name: customer_female, dtype: int64


Counts for customer_response:
0.0    12157
1.0     3572
Name: customer_response, dtype: int64


Counts for manager_response:
0.0    13904
1.0     1825
Name: manager_response, dtype: int64


Counts for representative_response:
1.0    14122
0.0     1607
Name: representative_response, dtype: int64


Counts for CEO_oversee:
1    11071
0     4658
Name: CEO_oversee, dtype: int64


Counts for county_LA:
0    10502
1     5227
Name: county_LA, dtype: int64


Counts for county_SF:
0    11218
1     4511
Name: county_SF, dtype: int64


Counts for county_OC:
0    12961
1     2768
Name: county_OC, dtype: int64


Counts for service_location_customer_home:
1    9569
0    6160
Name: service_location_customer_home, dtype: int64


Counts for service_location_business_address:
0    12686
1     3043
Name: service_location_business_address, dtype: int64


Counts for service_location_community_spaces:
0    14153
1     1576
Name: service_location_community_spaces, dtype: int64


Counts for service_location_apartments:
0    15400
1      329
Name: service_location_apartments, dtype: int64
```

Given the counts of each level of the categorical predictors, I have certain categories with very uneven distributions, such as service_location_apartments or race_other. For these features, setting min_samples_split too high might prevent each tree from splitting on these features, especially in deeper parts of the tree where the number of samples per node could naturally be lower. Thus, for categories with a smaller number of samples, I have to set a smaller min_samples_split to allow splits on these less frequent categories. 

A conservative starting point for min_samples_split could be around 5% to 10% of the smallest category size in the dataset. In this case, it is 329 from service_location_apartments. Multiplying this number with 0.05 results in ~16. I will use 15. This number would be a conservative start, ensuring that the model can still split on the smallest category. For max_depth, I will set the lowest number as the square root of the total number of features (n = 20) and will also use 1.0, which indicates 100% of the total features are used. For min_samples_leaf, I will use the same rule of thumb as with min_samples_split (~15). Since I have fewer features in the dataset, I may not have to worry about max_leaf_nodes and max_depth as it is unlikely that the model would overfit. Thus, I will include 'None' to not restrict those parameters along with other numbers. Below, I will test different thresholds, observing model performance through cross-validation to see if increasing or decreasing the parameters improves performance.

```ruby
# Create the parameter grid
param_grid = {
    'max_depth': [None, 5, 10], 
    'min_samples_split': [15, 50],
    'min_samples_leaf': [15, 50],
    'criterion': ['squared_error', 'absolute_error'], 
    'max_leaf_nodes': [20, 50],  
    'max_features': [1.0, 'sqrt'] 
}

# Setup the grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best score (neg MSE):", grid_search.best_score_)
```


<img width="983" alt="Screen Shot 2024-05-14 at 4 57 32 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/b3c50987-3e38-45d6-9b26-4cc6ad369678">

The results suggested neg MSE = -0.8845791452984593. The negative MSE indicates the mean squared error of the model, with a lower value representing better performance. Most machine learning libraries, including scikit-learn, have scoring functions designed to maximize a score. For many evaluation metrics, such as accuracy and precision, higher values are better . However, for metrics where lower values are better like mean squared error, the natural formulation conflicts with the maximization objective. Thus, Scikit-learn handles this by negating the scores of metrics where lower values are better. This way, the library can use the same optimization routines to maximize the (negative) score. The negative values are used during the optimization process, but when interpreting the results, we convert them back to positive values to understand the actual MSE. In other words, you can interpret negative MSE as a normal way to interpret a positive MSE. Thus, in this case, I have MSE = 0.88.

Notice that I did not integrate n_estimators in the GridSearchCV function. This is because setting the number of trees in the forest could significantly increase the computational complexity of the search function with marginal benefits. A better and more computationally efficient way is to manually test several values of n_estimators while keeping other parameters constant, according to the grid search suggestion, and observe how the model performance (e.g., accuracy or mean squared error) changes. You can also request out-of-bag (OOB) estimation to better estimate the optimal number of n_estimators by using the OOB as a validation set before testing the model on the actual test set.


```ruby
import matplotlib.pyplot as plt

# Possible values of n_estimators
n_estimators_values = [50, 100, 200, 300, 400, 500, 1000, 5000]
oob_errors = []

# Iterate over values of n_estimators
for n in n_estimators_values:
    model = RandomForestRegressor(n_estimators=n,
                                  max_depth= 20, max_features = 1.0, max_leaf_nodes = 50, min_samples_leaf= 15, 
                                  min_samples_split= 15,
                                  oob_score=True, n_jobs=-1, random_state=42, bootstrap=True)
    model.fit(X_train, y_train)
    # Record the OOB error
    oob_error = 1 - model.oob_score_  # oob_score_ gives the R^2 value, converting it to error
    oob_errors.append(oob_error)

# Plotting the OOB errors
plt.figure(figsize=(10, 5))
plt.plot(n_estimators_values, oob_errors, label='OOB Error')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('OOB Error')
plt.title('Effect of n_estimators on OOB Error')
plt.legend()
plt.show()
```

<img width="915" alt="Screen Shot 2024-05-14 at 8 33 44 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/e2b59e18-dc8c-4bd8-b1f4-14903e34533a">

The output shows that at n_estimators around 5000, the OOB error starts to stabilize and does not significantly decrease. Thus, I will use this number as utilizing a larger number could increase computational complexity without improving model performance. The OOB error is estimated by 1 - R squared, or the variance in the target variable explained by the model. According to the output, the OOB error is quite high (~0.81), indicating that my model does not fit the data well. The model performance could be worse when the model is used to predict the outcome of the test set. This could happen because RF might not be the best algorithm to describe the current data pattern. Let's fit the final model using this value of n_estimators along with fine-tuning other parameters based on the values that I obtained earlier to see my hypothesis is true.

```ruby
# Final model configuration 
final_model = RandomForestRegressor(n_estimators=5000,
                                    max_depth=10, max_features=1.0, max_leaf_nodes=50, min_samples_leaf=15, 
                                    min_samples_split=15,
                                    oob_score=True, n_jobs=-1, random_state=42, bootstrap=True)
final_model.fit(X_train, y_train)
```

```ruby
from sklearn.model_selection import cross_val_score

scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring='r2')
print(f'Cross-validated R^2 scores: {scores}')
print(f'Mean cross-validated R^2 score: {scores.mean()}')

```

Here is the output: 


<img width="746" alt="Screen Shot 2024-05-19 at 4 13 02 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/2f8ab264-89c6-499b-befa-a795f19bedd1">


The average R sqaured scores across five cross-validation folds are 0.096, suggesting that, on average, the model explains about 9.61% of the variance in the target variable. Now let's apply the model to the testing set to see its performance: 


```ruby
#test the model with the testing set 

from sklearn.metrics import r2_score

# Make predictions on the testing set
y_pred = final_model.predict(X_test)

# Calculate the R² score on the testing set
test_r2_score = r2_score(y_test, y_pred)

print(f'Test R² score: {test_r2_score}')

```

The Test R² score is 0.10267677698649125. The number is low and consistent with the performance of the training data. Note that the model performance of the testing set is slightly higher than the average performance of the training set. This is unusual and can occur due to several reasons. The first potential reason is model variance. In this case, I use 80% of the original data as the training set and 20% as the testing set. As the testing set is smaller, the R² score might be less stable and more susceptible to fluctuations due to the particular characteristics of the data. Samples in the testing set might be easier to predict or might be less diverse than the training data. A second reason could be that the model is underfitting or just poorly fitting. When it is applied to a new data set (testing set), the model fits better. However, note that despite the differences in performance between the training and testing sets, the difference is very small. The results across the validation folds of the training set also reflect that some folds' performance is quite close to the performance of the testing set. This indicates that the model is not likely underfitting or overfitting, and the discrepancy is likely because RF is not the right algorithm to explain the current dataset.

To confirm the hypothesis, I performed the following steps:

1) I re-split the training and testing set to have a ratio of 60:40 instead of 80:20 and repeated the same model building and evaluation stage that I described earlier. With this new ratio, the training data performance across the 5 validation folds is 0.091, with some folds' performance as close as the testing performance, which is 0.097. The findings indicate that the model is likely poorly fitting rather than an issue with the differences in distribution between the training and testing datasets.

2) As poor fitting could be a potential reason, I fine-tuned the hyperparameters again. This time, I did not limit the max_leaf_nodes and set the min_samples_split and min_samples_leaf as low as 2, which is the default setting.

<img width="999" alt="Screen Shot 2024-05-19 at 4 58 25 PM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/da081e55-d157-4a6c-9c57-1d0e3338c616">

According to the output, you can see that the model performance with both the testing and training sets improved overall from the previous parameter setting. However, the model performance is still quite low (approximately 15%). When you encounter this type of issue, other algorithms should be tested. However, for the purpose of the current demo, I will continue with RF. The next task to better understand the dataset is to test feature importance, which helps to identify which features contribute the most to the predictions.

```ruby
importances = final_model.feature_importances_
feature_names = X_train.columns
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), feature_names[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()
```

Here is the output: 

<img width="987" alt="Screen Shot 2024-05-20 at 9 49 10 AM" src="https://github.com/KayChansiri/demo_random_forest-/assets/157029107/8dff1582-6f56-4623-ace2-8958cf2dfdbb">


The y-axis represents the importance scores of the features, which are calculated based on how effectively each feature is used to split the data and reduce the model's error. Higher values indicate that a specific feature leads to larger gains in the purity of the nodes in the tree and a greater reduction in model error when that feature is used in a split. In this specific output, the representatives' years of experience has the highest importance score (approximately 0.5), indicating that this feature accounts for about 50% of the predictive power of the model. Note that the importance scores are typically normalized so that the sum of all feature importances equals 1. Each score can be interpreted as the proportion of the model's predictive power attributable to each feature.

Note that there is a significant drop in feature importance from the representatives' years of experience to the customers' age, which explains about 10% of the predictive power in the model. The next features in the output indicate progressively less importance. Overall, the findings suggest that the majority of the model's predictive capability is concentrated in a few features. The least five important features, including 'service_location_apartments', 'race_other', 'service_location_community_spaces', 'county_SF', and 'ethnicity_hispanic', each explain less than 0.05% of the predictive power. I will re-run the model and fine-tune hyperparameters based on the new data, excluding those features, to see if the model performance improves.


```ruby
# Separate features and target variable for training data
X_train = train_data.drop(columns=['satisfaction_rating', 'customer_id', 'feedback_phase', 'service_location_apartments', 'race_other', 'service_location_community_spaces', 'county_SF', 'ethnicity_hispanic']) #not plan to use ID for the analysis
y_train = train_data['satisfaction_rating']

# Separate features and target variable for testing data
X_test = test_data.drop(columns=['satisfaction_rating', 'customer_id', 'feedback_phase', 'service_location_apartments', 'race_other', 'service_location_community_spaces', 'county_SF', 'ethnicity_hispanic'])
y_test = test_data['satisfaction_rating']

# Define the model
rf = RandomForestRegressor(n_jobs=-1)

```

Performing the step above, the final R² score for the testing set is 0.1697135584517251, which is approximately 2% improved from the model when the least five important features were included. This indicates that excluding certain features with lower predictive powers may improve model performance. However, despite consistency in the model performance across the cross-validation folds of the training set and the testing set, the performance is still low, indicating that RF might not be the best algorithm for explaining customer satisfaction in the current dataset. Other ensemble techniques such as gradient boosting or ADA boosting should be explored.

Note that random forest is not the only ensemble technique that can be used. You can even achieve the best prediction by assembling different algorithms such as random forest, KNN, logistic regression, etc., all together. This method offers a diverse set of regressors or classifiers by using different training algorithms. This approach differs from the bagging of random forest, which uses the same training algorithm but trains them on different random subsets of the same training set. In addition to the bagging that I explained previously, there are also sampling techniques without replacement, known as pasting. However, I will not cover this topic for this demo. You can try pasting by setting bootstrap = False to see which ensemble method performs better.

It is also worth mentioning that although ensemble techniques are used to ensure good model performance, the method is less used in production because of the challenges in deployment and maintenance. However, for certain scenarios such as using RF to predict deposit subscriptions at a bank, the method is still popular in production as a small performance boost can lead to significant financial gain.


