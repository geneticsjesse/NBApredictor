# Full pipeline for ML model of a classification problem
# Dan Tulpan, dtulpan@uoguelph.ca
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from numpy import arange
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

# How to run:  python3  classification_example.py

# load dataset
url = "https://animalbiosciences.uoguelph.ca/~dtulpan/ansc6100/data/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# summarize data #
# -------------- #
print('Data summarization')
print('------------------')
# shape
print('\nDataset size: ', dataset.shape)

# head
print('\nFirst 10 lines of data:\n', dataset.head(10))

# descriptions
print('\nSummary stats of data:\n', dataset.describe())

# class distribution
print('\nClass distribution:\n', dataset.groupby('class').size())

# explore data visually #
# --------------------- #

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# histograms
dataset.hist()
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# compare algorithms #
# ------------------ #

# separate data into training/validation and testing datasets
array = dataset.values
X = array[:,:-1]
y = array[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# initialize models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate models
print('\nModel evalution - training')
print('--------------------------')
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# compare models based on training results
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison - before optimization')
pyplot.show()

# improve accuracy with hyper-parameter tuning
print('\nModel evaluation - hyper-parameter tuning')
print('-----------------------------------------')
model_params = dict()
model_params['LR'] = dict()
model_params['LR']['C'] = list(arange(0.1,2,0.1))
model_params['KNN'] = dict()
model_params['KNN']['n_neighbors'] = list(range(1,21,2))
model_params['DT'] = dict()
model_params['DT']['criterion'] = ['gini', 'entropy']
model_params['DT']['max_depth'] = list(range(1,10,1))
model_params['NB'] = dict()
model_params['NB']['var_smoothing'] = list(arange(1e-10,1e-08,1e-9))
model_params['SVM'] = dict()
model_params['SVM']['C'] = list(arange(0.01,1.6,0.2))

best_params = dict()
for name, model in models:
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    rand_search = RandomizedSearchCV(estimator=model, param_distributions=model_params[name], n_iter=5, n_jobs=-1, cv=cv, scoring='accuracy')
    rand_result = rand_search.fit(X_train, Y_train)
    print("Model %s -- Best: %f using %s" % (name, rand_result.best_score_, rand_result.best_params_))
    best_params[name] = rand_result.best_params_

# re-initialize models using best parameter settings
optimized_models = []
optimized_models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr', C=best_params['LR']['C'])))
optimized_models.append(('KNN', KNeighborsClassifier(n_neighbors=best_params['KNN']['n_neighbors'])))
optimized_models.append(('DT', DecisionTreeClassifier(criterion=best_params['DT']['criterion'], max_depth=best_params['DT']['max_depth'])))
optimized_models.append(('NB', GaussianNB(var_smoothing=best_params['NB']['var_smoothing'])))
optimized_models.append(('SVM', SVC(gamma='auto',C=best_params['SVM']['C'])))

print('\nModel evalution - optimized')
print('--------------------------')
results = []
names = []
for name, model in optimized_models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# compare optimized models based on training results
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison - after optimization')
pyplot.show()

# fit and save optimized models
for name, model in optimized_models:
	model.fit(X_train, Y_train)
	filename = name + '_optimized_model.sav'
	joblib.dump(model, filename)

# testing
print('\nModel testing')
print('-------------')
for name, model in optimized_models:
	model.fit(X_train, Y_train)
	predicted_results = model.predict(X_test)
	acc_result = accuracy_score(predicted_results,Y_test)
	print('%s accuracy: %f' % (name, acc_result))
	cm = confusion_matrix(predicted_results,Y_test)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm)
	disp.plot()
	pyplot.title('Confusion matrix corresp. to test results for ' + name)
	pyplot.xlabel('Ground truth')
	pyplot.ylabel('Predicted results')

	pyplot.show()

