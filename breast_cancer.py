from sklearn.datasets import load_breast_cancer
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
#setting path for Graphviz location
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
#Load Breast Cancer Dataset from sklearn.datasets
CancerDataset = load_breast_cancer()
dataFrame = pd.DataFrame(data=CancerDataset.data, columns=CancerDataset.feature_names)
dataFrame['target'] = CancerDataset.target
print("Breast Cancer Dataset:")
print(dataFrame.head())

#splitting the data into train dataset and test dataset
dataFrame_train, dataFrame_test = train_test_split(dataFrame, test_size=0.3)
dataFrame_train.shape, dataFrame_test.shape, dataFrame_train.shape[0] / (dataFrame_train.shape[0] + dataFrame_test.shape[0])


#Applying Randomforest classifier
rforest = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
rforest.fit(dataFrame_train.drop(columns='target'), dataFrame_train['target'])

print("Classification Report:")
print(classification_report(dataFrame_test['target'], rforest.predict(dataFrame_test.drop(columns='target'))))

def __input(dataFrame, exclude_columns=['target']):
    return dataFrame.drop(columns=list(set(exclude_columns) & set(dataFrame.columns)))

def __output(dataFrame, target_column='target'):
    return dataFrame[target_column]

def relable(dataFrame, model):
    dataFrame = dataFrame.copy()
    dataFrame['relabel'] = model.predict(__input(dataFrame))
    return dataFrame

# Relabelling Everything
dataFrame_train_tree = relable(dataFrame_train, rforest)
dataFrame_test_tree = relable(dataFrame_test, rforest)
dataFrame_tree = relable(dataFrame, rforest)

print(dataFrame_train_tree.head())

from sklearn.tree import DecisionTreeClassifier
from functools import partial
from sklearn.metrics import f1_score

__input = partial(__input, exclude_columns=['target', 'relabel'])
__rel = partial(__output, target_column='relabel')
__f1_score = partial(f1_score, average="macro")


dtree = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, min_impurity_decrease=0)
dtree.fit(__input(dataFrame_tree), __rel(dataFrame_tree))

print("Converting RandomForest to a Single Tree:")
print(f"Overfitting the relables (i.e. F1 score == 1.0) and have achieved RandomForest's behaviour perfectly!")
print(classification_report(__rel(dataFrame_train_tree), dtree.predict(__input(dataFrame_train_tree))))
assert __f1_score(__rel(dataFrame_train_tree), dtree.predict(__input(dataFrame_train_tree))) == 1.0

print("\n\n")
print(f"Depicts the performance on the actual `target` values of the test set")
print(classification_report(__output(dataFrame_test_tree), dtree.predict(__input(dataFrame_test_tree))))
assert __f1_score(__output(dataFrame_test), rforest.predict(__input(dataFrame_test_tree))) == __f1_score(__output(dataFrame_test), dtree.predict(__input(dataFrame_test_tree)))

from dtreeviz.trees import *
viz = dtreeviz(dtree,
               __input(dataFrame_tree),
               __output(dataFrame_tree),
               target_name='Cancer',
               feature_names=__input(dataFrame_test_tree).columns,
               class_names=CancerDataset.target_names.tolist()
              )
viz.view()





