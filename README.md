# ﻿Binary Classification using Gaussian Naive Bayes with the Diabetes Dataset 


## Introduction

Bayes's theorem describes the probability of an event, based on prior knowledge of conditons that might be related to the event. Bayes's theoren is stated as following:

![4211e3e7c3482573cdfbc0653d48a6279104c899](https://github.com/zypchn/Gaussian-Naive-Bayes/assets/144728809/1dd36f67-3a9b-4b92-bc04-82b82ed47dda) 
* ***P(A|B)*** is a conditional probability that is event A occurring given that B is true.
* ***P(B|A)*** is a conditional probability that is event B occurring given that A is true.
* ***P(A)*** and ***P(B)*** are the probabilities of observing A and B without any given conditions.

Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable. Given class variable ***y*** and feature vector ***x1*** through ***xn***, Bayes's theorem states the following equation:

![image](https://github.com/zypchn/Gaussian-Naive-Bayes/assets/144728809/88ff9203-f1f1-43c9-8907-e007eb3b7bf0)

Using the naive conditional independence assumption that ***P(xi|y,x1,...xi-1,xi+1,...,xn) = P(xi|y)*** for all ***i***, this relationship is simplified to: 
![Screenshot 2024-03-12 152808](https://github.com/zypchn/Gaussian-Naive-Bayes/assets/144728809/af566cbf-b07e-4531-a502-79a2d69a99bb)

Since ***P(x1,...,xn)*** is constant given the input, we can use the following classification rule:
![Screenshot 2024-03-12 153051](https://github.com/zypchn/Gaussian-Naive-Bayes/assets/144728809/d49e6395-210f-49b9-b110-a91986e269f3).

The different Naive Bayes classifiers differ mainly by the distributions (Gaussian, Multinomial, Bernoulli, Categorical, etc.) of features. 

In this project, Gaussian Naive Bayes algorithm was used because of the nature of features' values which is continous. When dealing with continuous data, a typical assumption is that the continuoıs values associated with each class are distributed according to a normal (Gaussian) distribution; hence the Gaussian Naive Bayes algorithm was used. Gaussian Naive Bayes algorithm states the following equation:
![Screenshot 2024-03-12 155722](https://github.com/zypchn/Gaussian-Naive-Bayes/assets/144728809/a6a729b1-540b-45f4-8e01-c18a5d3de0b3)


The dataset used was the Diabetes dataset containing 768 instances with 9 numeric features (including the class variable). The class variable is binary (0 or 1) to determine whether the patient has diabetes or not, so the task is to perform a binary classification.

The aim of this project was to train three seperate Gaussian Naive Bayes models and performing normalization and fine-tuning and explain the reasoning behind the results obtained. The following are the three models mentioned above:
**1)** A Gaussian Naive Bayes model without any normalization or fine-tuning.
**2)** A Gaussian Naive Bayes model with normalized the data using MinMaxScaler() from the Scikit Learn library.
**3)** A Gaussian Naive Bayes Model with normalized data using MinMaxScaler() and fine-tuned with changing the *var_smoothing* parameter. The optimal value for the *var_smoothing* parameter was searched using the GridSearchCV() algorithm in the Scikit Learn library.


## Methods

**Numerical values obtained from this project (accuracy score, confusion matrix) were discussed in the *Results* part of this report. The *Methods* part does not contain any numerical values (in terms of results), just explains the methods and how we used them to get the results we have.**

Language of this repository is listed as Jupyter Notebook thus, the language of our code is Python. Before diving into any piece of code we must import the necessary libraries. Main libraries used in this code are: *Scikit Learn (model training, tuning, evaluating)*, *Pandas (data manipulation)* and *Matplotlib (data visualization)*. The following are the classes and modules used in this project:
* GaussianNB from sklearn.naive_bayes
* GridSearchCV, train_test_split from sklearn.model_selection
* confusion_matrix, ConfusionMatrixDisplay, accuracy_score from sklearn.metrics
* MinMaxScaler from sklearn.preprocessing

The first step was to prepare the data before feeding it to the model. The Diabetes dataset was in a csv file so the initial move was to create a Pandas Data Frame called ***diabetes***, containing all the data. Then, the newly created dataset was partitioned into two datasets: ***X*** and ***y***. Data Frame ***X*** was obtained by dropping the *Outcome* column of the ***diabetes*** Data Frame, and Series ***y*** was obtained by keeping only the *Outcome* column. The following are the visualizations of ***X*** and ***y***, respectively:

<img src="https://github.com/zypchn/Gaussian-Naive-Bayes/assets/144728809/e7468bdb-d738-43b5-8e98-cab975ad5167" width="600">
<img src="https://github.com/zypchn/Gaussian-Naive-Bayes/assets/144728809/d954be33-8262-4f05-96f7-bfb19f079036" width="400">

Then, ***X*** and ***y*** was splitted into two datasets each, giving four datasets in the end: ***X_train***, ***X_test***, ***y_train***, ***y_test***. For this *train_test_split* function from the Scikit Library was used. Five parameters were given to the function: ***X***, ***y***, *random_state=42*, *test_size=0.2* and *shuffle=True*. *random_state* parameter controls the shuffle before the split with the given seed, *test_size* parameter controls the proportion of the dataset to include in the test split and *shuffle* controls whether the dataset should be shuffled or not; for better generilazation data should be shuffled before the split, hence *shuffle=True*.

After the creation of the datasets, next step was to create a model and feeding the data to it.
Our first model (and all the models in this project) was created with *GaussNB()* class mentioned above. We then feed our data to the model with *.fit()* method. We can also get the hyperparameters for the created model using *.get_params()* method. When used, *get_params()* returned the following dictionary: {'priors': None, 'var_smoothing': 1e-09}. These are the default hyperparameters of any instance of *GaussNB()* class. *priors* adjusts prior probabilities of the classes. Default value is *None* so the prior probabilities are computed based on the training set. If specified (with an array-like object), it states that certain classes are more likely than others, even if the training data suggests otherwise. *var_smoothing* is a positive float value that is used to add a small value to the variances of all features. This addition prevents the variance from being exactly zero, which can cause numerical instability issuies in the computations of the algorithm. This method is particularly useful when you have features with very low variance or when you have a small dataset (which is our case).
Then, we created a numpy array named ***y_pred*** which contains the predictions of our model. The predictions were made with *.predict()* method. We then computed our accuracy score using *accuracy_score()* method. *accuracy_score()* takes two arguments: the test data *y_test* and the prediction data *y_pred*. with *y_pred*, we can also compute a confusion matrix and display it with *ConfusionMatrixDisplay()* and *.plot()* method.

Our second model was trained with Normalized data and for that, *MinMaxScaler()* class was used. An instance of *MinMaxScaler()* class was created and named *scaler*. Then, a new train set was created with using the *.fit_transform()* method of our *scaler*. What *scaler* does is, it scales the dataset such that all feature values are in the range [0, 1]. The scaled ***x*** is computed with the following formula:

*X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))*

*X_scaled = X_std * (max - min) + min*

Same steps in the previous model (accuracy score and confusion matrix) were also applied in this model using the exact same methods but with small changes. We used *X_test_scaled*: a newly created dataset, obtained by scaling the *X_test* data.

Our third model was fine-tuned and trained with Normalized data. *X_train_scaled* was also used in this model. For fine-tuning *GridSearchCV()* was used to search the optimal *var_smoothing* hyperparameter amongst the values [9e-9, 3e-3, 7e-7, 5e-5]. A dictionary was defined named *params*: keys are the hyperparameters we want to search for and the values are arrays containing the values we want to try for the specified hyperparameter. We then created an instance of *GridSearchCV()* named *grid_search* and passed several parameters: *gaussNB, params, cv=3, scoring="accuracy", return_train_score=True*. First and second parameters are the estimator and the dictionary we want to use for testing given values, respectively. *cv* sets the cross-validation value, *scoring* sets the metric we want to evaluate our cross-validated model and *return_train_score* is to determine whether you want further insight about cross_validated models later on. What *GridSearchCV()* does is that it trains and evaluates a number of seperately cross-validated models and returns whichever scored the best according to the *scoring* parameter passed. The number of trained models can be determined using this formula: length of all arrays passed * cross-validation value. In our case we trained 4*3=12 models.

With *grid_search* we also used *.fit()* method using *X_train_scaled* and *y_train*. With *return_train_score* parameter set to True, we were able to see how each parameter performed with every cross-validation. These results were listed in an object named *cv_results_* (more specifficaly *grid_search.cv_results_*) and we were able to look up any of these results. We choose the *rank_test_score* array.
Finally, we created a new estimator named *best_estimator* using *grid_search.best_estimator_*. (parameter comparison and results will be discussed in the *Results* chapter). 

Same steps in the previous model (accuracy score and confusion matrix) were also applied in this model using the exact same methods.


## Results

### The Model with Default Hyperparamaters and Raw Data
The first model scored the highest accuracy compared to its other two counterparts. Despite that it was not fine-tuned and the data was not normalized, the model was still able to score an accuracy of 76.6%. Below is the confusion matrix for this model:

![b51221a1-941f-4a32-ae83-fc45bc9e88c3](https://github.com/zypchn/Gaussian-Naive-Bayes/assets/144728809/ea4b777b-1269-4318-a704-c0415e7a3acd)

As we can see the model classified 79 true-negatives and 39 true-positives, giving us 115 correctly classified instances in total. 

### The Model with Default Hyperparamaters and Normalized Data
The second model performed slightly worse compared to the first model. It scored an accuracy of 68.8% which is a drop of 7.8%. Below is the confusion matrix for this model: 

![c6013152-f806-4347-b2b0-402ade373ca1](https://github.com/zypchn/Gaussian-Naive-Bayes/assets/144728809/c9a3cbc9-88ad-453e-bb0a-4df71a1e97a1)

The model performed better with true-positives (with an increase of 7) but worse with true-negatives (with a decrease of 19). A similar case can be seen with false-positives (with an increase of 19), and false-negatives (with a decrease of 7).

### Fine-Tuned Model with Normalized Data
The third and last model performed exactly the same with the former one (both accuracy score and confusion marix), despite that it was fine-tuned using Grid Search. Amongst the values passed (9e-9, 3e-3, 7e-7, 5e-5), the best estimator had the value of 3e-3. An interesting thing was that models trained with the remaining values performed exactly or very similarly with each other, giving us an array of test score ranking [2, 1, 2, 2].
