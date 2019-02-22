import pandas as pd
from sklearn import tree
from treeviz import tree_print
from sklearn.metrics import accuracy_score


# Read the data
tennis_df = pd.read_csv('tennis_numeric.csv')
# Print the data
tennis_df


# Split the data into numerical and categorical
tennis_categorical = pd.DataFrame(tennis_df['play'])
tennis_numerical = tennis_df.drop(['play'], axis=1)

# Create the entropy tree
tennis_dtree = tree.DecisionTreeClassifier(criterion='entropy')

# Build the model
tennis_dtree.fit(tennis_categorical, tennis_numerical)

tree_print(tennis_dtree, tennis_categorical)