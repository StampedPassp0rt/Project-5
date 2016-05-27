
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics
from sqlalchemy import create_engine

engine = create_engine('postgresql://localhost:5432/titanic')

get_ipython().magic(u'matplotlib inline')


# ## Pre-Task: Describe the goals of your study

'''The goal is to construct a good model that classifies people in a disaster
as survivors or victims to better allocate resources and priorities for disaster
 relief. Technically speaking, we're determining probabilities of survival
 depending on characteristics of the people in the disaster area, and besides
 knowing these probabilities, we're assigning people to the survivor or not
 classes.

The data we'll use is from the sinking of the Titanic.'''

#Part 1: Acquire the Data

#Code to connect to the remote database...
'''
psql -h dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com -p 5432 -U dsi_student titanic
password: gastudents
'''

#1. Connect to the remote database

'''I am going to connect to the remote database and query it in the command line.'''

#2. Query the database and aggregate the data
'''Data in local db accessible via:
psql -h localhost -p 5432 titanic'''

## What is the proportion of survivors to non-survivors?
'''select sum(survived) from train;''' #342 survivors out of 891 passengers
print "There were %f proportion of survivors to non-survivors." % (float(342)/891)
'''So .383838 is the baseline for our model.'''

#What are some other exploratory questions?

'''We know this is Titanic passenger data, so exploring how the passengers are
distributed (men, women, type of cabin, etc. could be useful)'''

#Proportion of Male Passengers on Titantic
print "There were %f male passengers on the Titantic." % (float(577)/891)

#Proportion of Males who survived
'''select survived, count(sex) from train where sex = 'male' group by survived;'''
survivors = pd.read_sql_query('''select survived, sex, count(sex) from train group by survived, sex;''', engine)
survivors
male_prop_survive = survivors.ix[2][2]/float(sum(survivors['count'][survivors.sex == 'male']))
female_prop_survive = survivors.ix[1][2]/float(sum(survivors['count'][survivors.sex == 'female']))
female_prop_total = survivors.ix[1][2]/float(sum(survivors['count']))
male_prop_total = survivors.ix[2][2]/float(sum(survivors['count']))

print "Male Survivor Proportion of Male Passenger Base:\t", male_prop_survive
print "Female Survivor Proportion of Female Passenger Base:\t", female_prop_survive
print "Male Survivor Proportion of Total Passenger Base:\t", male_prop_total
print "Female Survivor Proportion of Total Passenger Base:\t", female_prop_total

#Obtaining all of Titantic data and putting into a pandas df

data = pd.read_sql('''select * from train;''', engine)
data.dtypes
#Distribution of Passenger classes
data.pclass.value_counts()

print data.cabin.value_counts()
'''The Cabin Data is very categorical and may be less valuable for insights than
 passenger class. In fact, it looks like the equivalent of room number.
 The potential insight may only be the level of the cabin, or the letter.'''
data.fare.plot(kind = 'hist', alpha = 0.3)
plt.title("Fares Distribution")

'''Is it possible to extract the level from Cabin?'''

data.cabin[data.cabin != None].count()

'''With only 204 cabin rows filled, imputation or matching by ticket to cabin
would be logical, but also a pain. Going to ignore the potential for data from cabin.'''


#3. What are the risks and assumptions of our data?

'''Assumptions:

1) We are assuming that the variables in the data are predictors of survival;
2) We are assuming that the socio-economic distribution of the Titanic (pclass)
is potentially meaningful and translatable to today (i.e. people making greater
than $x are more likely to survive in a future disaster)

Risks:
1) We know from history that the Titanic did not have enough lifeboats or an
orderly process for evacuation. Such process improvement may mean that using this
model on other disasters is less meaningful, even if we are being strict and looking
at sea-based disasters such as a sinking cruise ship.

2) The model may not be translatable to today's type of disasters.

3) Missing values and their imputation might actually skew results.

4) Multicollinearity/Correlation of certain variables (Fare and PClass)

5) Missing relational data - the SibSp and ParentChild count variables neglect
certain close familial relationships (close friend from a town; or mistress/fiancee).'''

# ## Part 2: Exploratory Data Analysis

# 1. Describe the Data

data.info()

'''Based on the above, gut is to not use cabin as a feature. As for age, what is misssing?'''

agenull = data[data.age.isnull()]
agenull

'''Dropping null ages...'''
newdata = data[data.age.notnull()]

print newdata.pclass.value_counts()


print newdata.sibsp.value_counts()

'''Plenty of passengers with no parents or children on board.'''

print newdata.parch.value_counts()
# #### 2. Visualize the Data

#Hist of age

import seaborn as sns
sns.pairplot(newdata)

#Plot of age to survival
'''Not much to see, a slight skew towards youth for survival.'''
plt.scatter(newdata.survived, newdata.age)

plt.figure()

newdata.groupby('survived').pclass.value_counts().plot(kind = 'bar', alpha = 0.3)

plt.title('Survived by Passenger Class')
plt.show()
# In[ ]:




# In[ ]:




# ## Part 3: Data Wrangling

# #### 1. Create Dummy Variables for *Sex*

newdata.dtypes

'''Used patsy to create dummies for Sex. Wonder if pclass should be a dummy.
from patsy import dmatrices

y, X = dmatrices("survived ~ pclass + sex + age + sibsp + parch + fare", newdata, return_type = "dataframe")
'''



'''Changed mind, regular get dummies for Sex...'''

dummies = pd.get_dummies(newdata[['sex', 'pclass']], drop_first = True)
dummies.head()
newdata.sex
features = ['age', 'sibsp', 'parch', 'fare']
X = newdata[features].join(dummies)
X
y = newdata.survived

# ## Part 4: Logistic Regression and Model Validation

# #### 1. Define the variables that we will use in our classification analysis

from sklearn.tree import DecisionTreeClassifier

dtbasic = DecisionTreeClassifier(random_state = 31, class_weight = 'balanced')

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
'''Going to see if we can find optimal parameters for tree and from there the features.'''
tree_params = {'criterion': ['entropy', 'gini'], 'splitter': ['random', 'best'],
                'max_depth': [2,3,4,5,6], 'min_samples_split' : [10,30,50], 'min_samples_leaf' : [5, 10]}

gsdtbasic = GridSearchCV(dtbasic,tree_params, cv = KFold(len(y), n_folds = 5, shuffle = True), n_jobs = -1, verbose = True)
gsdtbasic.fit(X,y)
print gsdtbasic.best_estimator_
X.columns

gsdtbasic.best_score_
feature_importance = pd.DataFrame(gsdtbasic.best_estimator_.feature_importances_, columns = ['importance'], index = X.columns)
feature_importance.sort_values('importance', ascending = False)

dtbasic.fit(X,y)

feature_importance_basic = pd.DataFrame(dtbasic.feature_importances_, columns = ['importance'], index = X.columns)
feature_importance_basic.sort_values('importance', ascending = False)

'''Note that with GridSearched tree, sibsp and parch fall out.
So I'll use the features from the grid-searched tree.'''

#Train Test Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .25, random_state = 31, stratify = y)

X_train.head()

'''Does the decision tree change with training only data?'''
#Grid Search on best decision tree
dtbasic_train = DecisionTreeClassifier(random_state = 31, class_weight = 'balanced')

gsdtbasic_train = GridSearchCV(dtbasic_train,tree_params, cv = KFold(len(y_train), n_folds = 5, shuffle = True), n_jobs = -1, verbose = True)
gsdtbasic_train.fit(X_train,y_train)
print gsdtbasic.best_estimator_
print gsdtbasic.best_score_
print "Accuracy of best estimator for Gridsearched Decision Tree on test data:\t", metrics.accuracy_score(y_test, gsdtbasic.predict(X_test))

feature_importance_train = pd.DataFrame(gsdtbasic_train.best_estimator_.feature_importances_, columns = ['importance'], index = X.columns)
feature_importance_train.sort_values('importance', ascending = False)

#Fitting on basic decision tree for training data

dtbasic_train.fit(X_train,y_train)
y_pred_dt_train = dtbasic_train.predict(X_test)
print "Accuracy for basic decision tree on when training on training data:\t", metrics.accuracy_score(y_test, y_pred_dt_train)

feature_importance_basic_train = pd.DataFrame(dtbasic_train.feature_importances_, columns = ['importance'], index = X.columns)
feature_importance_basic_train.sort_values('importance', ascending = False)

'''I believe that even when using a decision tree for feature selection, we want
to consider a train and test set. Looking above, when using all of X and y, we tossed
out sibsp and parch for our GridSearch Decision Tree. When just using a training set,
we kept those in, but tossed out fare. Intuititively, that makes sense since fare
should be highly correlated with Passenger Class.

Hence, for our logistic model, I will use an initial feature set from the Grid Search best estimator
that is on the training set, which keeps in five of the six features I used.'''

#Putting together a scores dictionary...
test_accuracy_scores = {}
test_accuracy_scores['gsdt'] = metrics.accuracy_score(y_test, gsdtbasic.predict(X_test))
test_accuracy_scores['dt'] = metrics.accuracy_score(y_test, y_pred_dt_train)
test_accuracy_scores

# #### 2. Transform "Y" into a 1-Dimensional Array for SciKit-Learn

'''Done above...'''




# #### 3. Conduct the logistic regression - using training data

lr = LogisticRegression(random_state = 31)
lr_trim_features = LogisticRegression(random_state = 31)
#Fitting on all the features - male, age, fare, pclass, sibsp, parch
lr.fit(X_train,y_train)

#Fitting on grid searched decision tree features.
grid_features = ['pclass', 'age', 'fare', 'sibsp', 'parch']
lr_trim_features.fit(X_train[grid_features], y_train)


# #### 4. Examine the coefficients to see our correlations
#Coefficients for all features....
full_features_coef_lr = pd.DataFrame(lr.coef_, columns = X_train.columns)
full_features_coef_lr
#Coefficients for trimmed logistic from Grid Search Decision Tree features
pd.DataFrame(lr_trim_features.coef_, columns = X_train[grid_features].columns)


# #### 6. Test the Model by introducing a *Test* or *Validaton* set

#Testing on the grid features...

'''Did a train test split above...'''

# #### 7. Predict the class labels for the *Test* set
#y_pred for full Logistic
y_pred = lr.predict(X_test)

#y_pred for trimmed Logistic
y_pred_trim = lr_trim_features.predict(X_test[grid_features])



# #### 8. Predict the class probabilities for the *Test* set
#Full logistic
y_pred_proba = lr.predict_proba(X_test)

#Trimmed Logistic
y_pred_proba_trim = lr_trim_features.predict_proba(X_test[grid_features])



# #### 9. Evaluate the *Test* set
print "Accuracy of the Full logistic on test data is:\t", metrics.accuracy_score(y_test, y_pred)
print "Accuracy on the testing data with the grid-searched DT features is:\t", metrics.accuracy_score(y_test, y_pred_trim)
'''Note two things: 1) the Full Logistic has a better accuracy score than the trimmed one;
2) The Grid Search Decision Tree has a better accuracy score.'''
test_accuracy_scores['logistic_full'] = metrics.accuracy_score(y_test, y_pred)
test_accuracy_scores['logistic_trim'] = metrics.accuracy_score(y_test, y_pred_trim)


# #### 10. Cross validate the test set
from sklearn.cross_validation import StratifiedKFold
scores = cross_val_score(lr, X_test, y_test, cv = StratifiedKFold(y_test, n_folds = 5, shuffle = True, random_state = 31))
print "Mean cross-validated score on test set is:\t", scores.mean()


# #### 11. Check the Classification Report
'''Note, since the Full Logistic has a better accuracy score, using it for the classification report.'''
print metrics.classification_report(y_test, y_pred)


# #### 12. What do the classification metrics tell us?

'''The Classification metrics tell us that we are more precise at predicting survival.
So we are decent at filtering out negatives. However, our recall for survival is not great.
And if for disasters, we want to minimize our false negatives, we would want our recall
to be much better for our survival.

So our model is not good at recall, or getting the True Positive Rate to 100%.'''

# #### 13. Check the Confusion Matrix

cm_logistic = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred), index = ['No Survival', 'Survived'], columns = ['Pred No Survived', 'Pred Survived'])

cm_logistic


# #### 14. What does the Confusion Matrix tell us?

'''Our model has a higher proportion of false negatives to positives than false positives to negatives.
The model handles negatives better. This actually is disturbing because in a disaster,
all else being equal, we would probably want to err on overestimating the survivors (false positives)
and minimizing false negatives. The headlines would probably be more oriented towards government waste
rather than government tragedy (deaths that could have been prevented due to being prepared for a
number of survivors), which is preferable.'''

# #### 15. Plot the ROC curve

Y_score_lr = lr.decision_function(X_test)

FPR_logistic = dict() #false positive rate. X-axis for ROC Curve
TPR_logistic = dict() #true positive rate. Y-axis for ROC curve
ROC_AUC = dict()

FPR_logistic[1], TPR_logistic[1], thresholds_logistic = metrics.roc_curve(y_test, Y_score_lr)
ROC_AUC[1] = metrics.auc(FPR_logistic[1], TPR_logistic[1])

plt.figure()
plt.plot(FPR_logistic[1], TPR_logistic[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Titanic data - Logistic Regression on Six Features')
plt.legend(loc="lower right")
plt.show()



# #### 16. What does the ROC curve tell us?
'''The ROC curve is showing us for the Logistic Regression, the cost of pushing
up our True Positive Rate to be more prepared for a disaster recovery situation
similar to the Titanic. If we pushed our False Positive Rate to 80%, we would be
near 100% for our True Positive Rate, and at 100% if our FPR is 85%.'''

# ## Part 5: Gridsearch

# #### 1. Use GridSearchCV with logistic regression to search for optimal parameters
#
# - Use the provided parameter grid. Feel free to add if you like (such as n_jobs).
# - Use 5-fold cross-validation.
# Also doing on the training set....


logreg_parameters = {
    'penalty':['l1','l2'],
    'C':np.logspace(-5,1,50),
    'solver':['liblinear']
}
grid_lr = GridSearchCV(lr, param_grid = logreg_parameters, cv = KFold(len(y_train), n_folds = 5, shuffle = True), n_jobs = -1, verbose = True)
grid_lr.fit(X_train, y_train)

# #### 2. Print out the best parameters and best score. Are they better than the vanilla logistic regression?

print "Best Estimator for Logistic Regression from Grid Search:\t", grid_lr.best_estimator_
print ""
print "Best Parameters for LR:\t", grid_lr.best_params_
print "Best Score:\t", grid_lr.best_score_

#Predict on test data
y_pred_grid_lr = grid_lr.predict(X_test)
print "Accuracy Score for Grid LR:\t", metrics.accuracy_score(y_test, y_pred_grid_lr)
test_accuracy_scores['grid_logistic'] = metrics.accuracy_score(y_test, y_pred_grid_lr)


# #### 3. Explain the difference between the difference between the L1 (Lasso) and L2 (Ridge) penalties on the model coefficients.

'''L1 forces a sparcer coefficient set because its use of absolute value of the
coefficient in the loss function essentially results in a model that contorts itself
less to outliers.

L2 is more sensitive to outliers, and usually as a result keeps more of the coefficients in.
In the case above, our estimator used the L2 penalty.'''

# #### 4. What hypothetical situations are the Ridge and Lasso penalties useful?

'''Lasso might be useful in two situations:

1) When the number of features is large, computational time matters, and hence
sparseness of coefficients is preferred;
2) When you suspect you want a less complex model due to outliers.

Ridge is useful to reduce complexity while still keeping in many features.'''

# #### 5. [BONUS] Explain how the regularization strength (C) modifies the regression loss function. Why do the Ridge and Lasso penalties have their respective effects on the coefficients?

'''Regularization strength also tells the logistic regression how tight or loose to be
when fitting to data.

A small C is very strict in fit, and hence pushes towards reducing the complexity of the model.

In short, it can reinforce sparsity when very small (less than 1) and with L1.
It can push towards sparsity or smaller coefficients when very small and with Ridge.
If uncertain how tight a fit you want, a big C (probably greater than 100) is preferable.'''


# #### 6.a. [BONUS] You decide that you want to minimize false positives.
#Use the predicted probabilities from the model to set your threshold for
#labeling the positive class to need at least 90% confidence.
#How and why does this affect your confusion matrix?
'''Hypotetically, by minimizing false positives, I am pushing for a low false positive rate.
This would end up reducing my true positive rate also when looking at the ROC curve.
'''

'''Just an FYI that age is a vastly different scale than the rest of the variables.
I am showing the plot and considering scaling it.'''

newdata.age.plot(kind = 'hist', alpha = .3)

#Scaling age and fare.

from sklearn.preprocessing import RobustScaler

X_scaled = RobustScaler().fit_transform(X[['age', 'fare']])
X_scaled = pd.DataFrame(X_scaled, columns = ['age', 'fare'], index = X.index)
#join with rest of Data

X_scaled = X_scaled.join(dummies)
X_scaled = X_scaled.join(X[['sibsp', 'parch']])
X.info()

#Train Test Split on Scaled Data...

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size = .25, stratify = y, random_state = 31)

#Grid Search Logistic regression
from sklearn.cross_validation import StratifiedKFold
grid_lr_scaled = GridSearchCV(lr, logreg_parameters, cv = StratifiedKFold(y_train, n_folds = 5, shuffle = True), n_jobs = -1, verbose = 1)
grid_lr_scaled.fit(X_train_scaled, y_train)

print grid_lr_scaled.best_estimator_
print grid_lr_scaled.best_params_

#Predict on best LR regressor
y_pred_lrscaled_grid = grid_lr_scaled.predict(X_test_scaled)
print metrics.accuracy_score(y_test, y_pred_lrscaled_grid)
test_accuracy_scores['grid_lr_scaled'] = metrics.accuracy_score(y_test, y_pred_lrscaled_grid)

cm_logistic_scaled = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_lrscaled_grid), columns = ['pred_no_survive', 'pred_survive'], index = ['no_survive', 'survive'])
# ## Part 6: Gridsearch and kNN

# #### 1. Perform Gridsearch for the same classification problem as above,
#but use KNeighborsClassifier as your estimator
#
# At least have number of neighbors and weights in your parameters dictionary.

from sklearn.neighbors import KNeighborsClassifier

knn_params = {'n_neighbors' : range(1,20), 'weights' : ['uniform', 'distance']}

knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid = knn_params, cv = StratifiedKFold(y_train, n_folds = 5, shuffle = True), n_jobs = -1, verbose = True)

knn_grid.fit(X_train_scaled, y_train)





# #### 2. Print the best parameters and score for the gridsearched kNN model.
#How does it compare to the logistic regression model?

print "Best Parameters for KNN are:\t", knn_grid.best_params_
print "Best Estimator for KNN is:\t", knn_grid.best_estimator_
print "Best Score for KNN is:\t", knn_grid.best_score_

#Predict on testing data.
y_pred_knn = knn_grid.predict(X_test_scaled)
print "Accuracy Score for knn_grid is:\t", metrics.accuracy_score(y_test, y_pred_knn)
test_accuracy_scores['knn_grid'] = metrics.accuracy_score(y_test, y_pred_knn)

'''The accuracy score for the knn is marginally better than my logistic regression
on scaled data. See below for scores collected to date.'''

test_accuracy_scores


# #### 3. How does the number of neighbors affect the bias-variance tradeoff of your model?
'''The greater the number of neighbors, the more of a risk that the model is overfitted.
So greater number of neighbors is equivalent to attempting to reduce variance in our training set
at the cost of bias when going to the test set. Interestingly, looking above,
the best score for KNN on the grid search is 0.8, and the accuracy score on my holdout
is .83. So we have mitigated for the bias-variance tradeoff.'''

# #### [BONUS] Why?

# #### 4. In what hypothetical scenario(s) might you prefer logistic regression
#over kNN, aside from model performance metrics?

'''Logistic Regression is superior to KNN when I want a parametric model that
is not a black box (i.e. I plug in values like a linear regression to get a
probability prediction).

Logistic Regression is also good when the model's features are likely linear in relationship.

Logistic Regression also gives coefficients with interpretable output and p-values
to better tell us which coefficients are important.'''


# #### 5. Fit a new kNN model with the optimal parameters found in gridsearch.

knn_grid.best_estimator_.fit(X_train_scaled, y_train)




# 6. Construct the confusion matrix for the optimal kNN model.
# Is it different from the logistic regression model? If so, how?
'''replicating what I did above...'''
y_pred_knngrid = knn_grid.best_estimator_.predict(X_test_scaled)
metrics.accuracy_score(y_test, y_pred_knngrid)

knn_cm = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_knngrid), columns = ['pred_no_survived', 'pred_survived'], index = ['no_survived', 'survived'])
knn_cm
'''KNN Confusion Matrix'''

'''Logistic Confusion Matrix'''
cm_logistic_scaled
'''Our false positive rate declined for KNN and True Positive Rate also decreased
marginally.'''

# #### 7. [BONUS] Plot the ROC curves for the optimized logistic regression model
#and the optimized kNN model on the same plot.

def grid_fxn(model, params):
    gridfxn = GridSearchCV(model, params, cv = StratifiedKFold(y_train, n_folds = 5, shuffle = True), n_jobs = -1, verbose = True)
    gridfxn.fit(X_train_scaled, y_train)
    print "Best Estimator:\t", gridfxn.best_estimator_
    print "Best Score:\t", gridfxn.best_score_
    #print "Best Params:\t", gridfxn.best_params_
    y_pred_grid = gridfxn.predict(X_test_scaled)
    print "Accuracy Score for Grid is:\t", metrics.accuracy_score(y_test, y_pred_grid)
    test_accuracy_scores[model] = metrics.accuracy_score(y_test, y_pred_grid)
    cm = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_grid), columns = ['pred_no_survived', 'pred_survived'], index = ['no_survived', 'survived'])
    return cm




# ## Part 7: [BONUS] Precision-recall

# #### 1. Gridsearch the same parameters for logistic regression but change the scoring function to 'average_precision'
#
# `'average_precision'` will optimize parameters for area under the precision-recall curve instead of for accuracy.

# In[ ]:




# #### 2. Examine the best parameters and score. Are they different than the logistic regression gridsearch in part 5?

# In[ ]:




# #### 3. Create the confusion matrix. Is it different than when you optimized for the accuracy? If so, why would this be?

# In[ ]:




# #### 4. Plot the precision-recall curve. What does this tell us as opposed to the ROC curve?
#
# [See the sklearn plotting example here.](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

# In[ ]:




# ## Part 8: [VERY BONUS] Decision trees, ensembles, bagging

# #### 1. Gridsearch a decision tree classifier model on the data, searching for optimal depth.
# Create a new decision tree model with the optimal parameters.

dt_new = DecisionTreeClassifier(random_state = 31, class_weight = 'balanced')
tree_params = {'criterion': ['entropy', 'gini'], 'splitter': ['random', 'best'],
                'max_depth': [2,3,4,5,6], 'min_samples_split' : [10,30,50], 'min_samples_leaf' : [5, 10]}

grid_fxn(dt_new, tree_params)

test_accuracy_scores

# #### 2. Compare the performace of the decision tree model to the logistic regression and kNN models.

cm_logistic_scaled

knn_cm

'''The decision tree has a similar accuracy to the logistic with scaled data,
but is slightly worse than the knn. Looking at the confusion matrices, Logistic
performs better if we want to have a good True Positive Rate out of the tree.'''


# #### 3. Plot all three optimized models' ROC curves on the same plot.

# In[ ]:




# #### 4. Use sklearn's BaggingClassifier with the base estimator
#your optimized decision tree model. How does the performance compare to the single decision tree classifier?

from sklearn.ensemble import BaggingClassifier


griddt = GridSearchCV(dt_new, tree_params, cv = StratifiedKFold(y_train, n_folds = 5, shuffle = True), n_jobs = -1, verbose = True)
griddt.fit(X_train_scaled, y_train)
print "Best Estimator:\t", griddt.best_estimator_
print "Best Score:\t", griddt.best_score_
print "Best Parameters for dt are:\t", griddt.best_params_


baggingdt = BaggingClassifier(griddt.best_estimator_, random_state = 31)
baggingdt.fit(X_train_scaled, y_train)
y_pred_bagging = baggingdt.predict(X_test_scaled)
test_accuracy_scores['baggingdt_scaled'] = metrics.accuracy_score(y_test, y_pred_bagging)
test_accuracy_scores['baggingdt_scaled']

'''The performance is better, marginally...'''

test_accuracy_scores
# #### 5. Gridsearch the optimal n_estimators, max_samples, and max_features for the bagging classifier.

bagging_params = {'n_estimators': [5,10,15,20,30], 'max_samples' : [.5, .75, 1], 'max_features': [.5, .75, 1]}

bagging_grid = GridSearchCV(baggingdt, bagging_params, cv = StratifiedKFold(y_train, n_folds = 5, shuffle = True), n_jobs = -1, verbose = True)
bagging_grid.fit(X_train_scaled, y_train)
print "Bagging Grid Best Params:\t", bagging_grid.best_params_
print "Bagging Grid Best Score:\t", bagging_grid.best_score_
print "Bagging Grid Best Estimator:\t", bagging_grid.best_estimator_



# #### 6. Create a bagging classifier model with the optimal parameters and compare it's performance to the other two models.

y_pred_bagginggriddt = bagging_grid.predict(X_test_scaled)
test_accuracy_scores['bagging_grid_dt'] = metrics.accuracy_score(y_test, y_pred_bagginggriddt)
test_accuracy_scores['bagging_grid_dt']

'''Accuracy went down compared to the other models...'''

cm_bagginggriddt = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_bagginggriddt), columns = ['pred_no_survived', 'pred_survived'], index = ['no_survived', 'survived'])
cm_bagginggriddt

cm_bagging_dt = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_bagging), columns = ['pred_no_survived', 'pred_survived'], index = ['no_survived', 'survived'])
cm_bagging_dt

cm_logistic_scaled

knn_cm

grid_lr_scaled.best_estimator_.coef_

grid_lr_scaled.best_estimator_.intercept_
