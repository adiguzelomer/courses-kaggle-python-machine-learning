"""
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")
"""



# Import the Numpy library
import numpy as np

# Import 'tree' from scikit-learn library
from sklearn import tree


#------------------------------------------------------------------------------------------
print
print("------------------------------------------ Exercise 2: Cleaning and Formatting your Data------------------------------------------------")

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

#Print the Sex and Embarked columns
print(train["Sex"])
print(train["Embarked"])

#------------------------------------------------------------------------------------------
print
print("------------------------------------------ Exercise 3: Creating your first decision tree------------------------------------------------")

# Print the train data to see the available features
print(train)

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one,target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))


#------------------------------------------------------------------------------------------
print
print("------------------------------------------ Exercise 4: Interpreting your decision tree------------------------------------------------")

print(my_tree_one.feature_importances_)

#------------------------------------------------------------------------------------------
print
print("------------------------------------------ Exercise 5: Predict and submit to Kaggle------------------------------------------------")

# Impute the missing value with the median
test.Fare[152] = test.Fare.median()

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex","Age", "Fare"]].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)
print(my_prediction)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

#------------------------------------------------------------------------------------------
print
print("------------------------------------------ Exercise 6: Overfitting and how to control it------------------------------------------------")

# Create a new array with the added features: features_two
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth =10, min_samples_split =5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two,target)

#Print the score of the new decison tree
print(my_tree_two.score(features_two,target))

#------------------------------------------------------------------------------------------
print
print("-------------------------------- Exercise 7: Feature-engineering for our Titanic data set--------------------------------------------")

# Create train_two with the newly defined feature
train_two = train.copy()
train_two["family_size"] =train["SibSp"] + train["Parch"] + 1

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch","family_size"]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three,target)

# Print the score of this decision tree
print(my_tree_three.score(features_three, target))