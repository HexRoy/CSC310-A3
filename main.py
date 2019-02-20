import pandas as pd
from sklearn import tree
from treeviz import tree_print
from sklearn.metrics import accuracy_score


tennis_df = pd.read_csv('tennis_numeric.csv')
tennis_df

tennis_dtree = tree.DecisionTreeClassifier(criterion='entropy')
#tennis_dtree.fit(features_df)
