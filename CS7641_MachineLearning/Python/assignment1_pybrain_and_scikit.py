# Import various helpful libraries
import numpy as np
from scipy import diag
import matplotlib.pyplot as plt
import pydot

"""
Requirements
 - scikit-learn
 - pybrain
 - matplotlib
 - pydot
 - GraphViz
 - libsm (http://www.lfd.uci.edu/~gohlke/pythonlibs/#libsvm)
 - numpy
 - scipy

--------------------
Pros:
 - Between these two modules, you cover a wide array of ML algorithms
 - Excellent integration with other Python science libraries (numpy and scipy)
 - Since it's mostly in Python, you can dive into the code to see what the algorithms are doing
 - The above is very useful in debugging as well

Cons:
 - PyBrain doesn't cover decision trees, and scikit-learn doesn't cover neural networks. So you'll have to use both
 - PyBrain's documentation isn't great.
 - scikit-learn's processes are more manual than other systems (see tree cross validation)

"""
from sklearn.cross_validation import KFold
def scikitFolds(data, target, learners, folds):
    """
    Compute a k-folds validation on a scikit-learn data set for several models
    """
    learner_err = []
    for model in learners:
        err = []
        kf = KFold(len(target), n_folds=folds, shuffle=True)
        for fold, breakdown in enumerate(kf):
            train, test = breakdown

            trn_data = data[train]
            tst_data = data[test]
            trn_target = target[train]
            tst_target = target[test]
            clf = model.fit(trn_data, trn_target)
            err.append(clf.score(tst_data, tst_target))
        learner_err.append(sum(err) / len(err))
    return learner_err

# #################################################################################
# Create a set of data points in three different classes.
# Distribute the points randomly about some XY mean with some covariance
N = 100
means = [(-1, 0), (2, 4), (3, 1)]
cov = [diag([1, 1]), diag([0.5, 1.2]), diag([1.5, 0.7])]
names = ['T1', 'T2', 'T3']

np.random.seed(25)  # For consistency
data = []
for n in xrange(N):
    for ii, name in enumerate(names):
        x, y = np.random.multivariate_normal(means[ii], cov[ii])
        data.append([x, y, name])

##################################################################################
# Plot the data

# This is a little complicated because we're plotting each point individually,
# so we can't use the standard ax.legend command. It would be a little simpler
# if we sorted by type first, then used the scatter command to plot them.
fig_explore = plt.figure()
ax = fig_explore.add_subplot(1, 1, 1)
ax.set_title("Random Data")
ax.set_xlabel('X')
ax.set_ylabel('Y')
for entry in data:
    x = entry[0]
    y = entry[1]
    name = entry[2]

    marker = ''
    if name == names[0]: marker = 'r*'
    if name == names[1]: marker = 'go'
    if name == names[2]: marker = 'k^'
    ax.plot(x, y, marker)

# Get artists and labels for legend and chose which ones to display
handles, labels = ax.get_legend_handles_labels()
display = (0, 1, 2)

# Create custom artists
t1Artist = plt.Line2D((0, 0), (0, 0), color='r', marker='*', linestyle='')
t2Artist = plt.Line2D((0, 0), (0, 0), color='g', marker='o', linestyle='')
t3Artist = plt.Line2D((0, 0), (0, 0), color='k', marker='^', linestyle='')

#Create legend from custom artist/label lists
ax.legend([handle for i, handle in enumerate(handles) if i in display] + [t1Artist, t2Artist, t3Artist],
          [label for i, label in enumerate(labels) if i in display] + ['T1', 'T2', 'T3'])

# Show our distributions
#plt.show()

##################################################################################
# Create two data storage mechanisms. One for PyBrain and one for scikit-learn.
# We'll have to use both to cover all the analysis methods.

# We have to convert the data labels to integers
targets = []
for instance in data:
    if instance[2] == 'T1':
        targets.append(0)
    elif instance[2] == 'T2':
        targets.append(1)
    elif instance[2] == 'T3':
        targets.append(2)

# scikit data sets can be kept as basic numpy arrays. For this, I'm going to create a Bunch object
from sklearn.datasets.base import Bunch

sk_data = Bunch()
#sk_data['data'] = np.vstack((np.array([d[0] for d in data]), np.array([d[1] for d in data]))).T
sk_data['data'] = data[:,:-1] # this will return the data array without the last column , looks simpler than above and saves on the transposing operation too.
sk_data['feature_names'] = ['X', 'Y']
sk_data['target'] = np.array(targets)
sk_data['target_names'] = ['T1', 'T2', 'T3']

# PyBrain data sets are stored in specific containers
from pybrain.datasets import ClassificationDataSet

pb_data = ClassificationDataSet(2, 1, nb_classes=3)
for ii in xrange(len(data)):
    pb_data.addSample(np.array([data[ii][0], data[ii][1]]), [targets[ii]])

##################################################################################
# Split into test and train data.

# Create a random split of the data indexes. I stole this from the code of
# PyBrain's SupervisedDataSet.splitWithProportion (more on that below)
Ndata = len(data)
proportion = 0.25
indices = np.random.permutation(Ndata)
separator = int(Ndata * proportion)
left_indices = indices[:separator]
right_indices = indices[separator:]

# Split the scikit-learn data
sk_trn = Bunch()
sk_trn['data'] = sk_data.data[right_indices]
sk_trn['feature_names'] = sk_data.feature_names
sk_trn['target'] = sk_data.target[right_indices]
sk_trn['target_names'] = sk_data.target_names

sk_tst = Bunch()
sk_tst['data'] = sk_data.data[left_indices]
sk_tst['feature_names'] = sk_data.feature_names
sk_tst['target'] = sk_data.target[left_indices]
sk_tst['target_names'] = sk_data.target_names

# Split the PyBrain data. I suspect there's a better way to do this, but I haven't found it.
#
# In theory, ClassificationDataSet has the function splitWithProportion to
# do just this. However, that returns the parent class SupervisedDataSet,
# so it doesn't work very well. I'm sure there's a simple way to convert
# it back, but I don't know it.
pb_trn = ClassificationDataSet(2, 1, nb_classes=3)
pb_tst = ClassificationDataSet(2, 1, nb_classes=3)
for ii in xrange(len(pb_data)):
    xy, cls = pb_data.getSample(ii)
    if ii in right_indices:
        pb_trn.addSample(xy, cls)
    else:
        pb_tst.addSample(xy, cls)

if not len(pb_trn) == len(sk_trn.data):
    raise "PyBrain and scikit-learn have different numbers of training samples"
if not len(pb_tst) == len(sk_tst.data):
    raise "PyBrain and scikit-learn have different numbers of testing samples"
if len(pb_trn) < len(pb_tst):
    raise "Training on fewer than half of the samples. Please remove this check if this is deliberate."

print "\n" + "*" * 50
print "Number of data points      : " + str(len(pb_data))
print "Number of training cases   : " + str(len(pb_trn))
print "Number of test cases       : " + str(len(pb_tst))
print "Input and output dimensions: ", pb_trn.indim, pb_trn.outdim


##################################################################################
# Create a decision tree (scikit-learn only)
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(sk_trn.data, sk_trn.target)

# Write the data using GraphVis and pydot
from sklearn.externals.six import StringIO

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("scikit_tree.pdf")

# Check the accuracy
print "\n" + "*" * 50
print "DEFAULT TREE"
print "Training Accuracy: " + str(clf.score(sk_trn.data, sk_trn.target))
print "Testing  Accuracy: " + str(clf.score(sk_tst.data, sk_tst.target))

# K-Fold validation
learners = []
learners.append(tree.DecisionTreeClassifier())
learners.append(tree.DecisionTreeClassifier(min_samples_split=20))
learners.append(tree.DecisionTreeClassifier(min_samples_leaf=5))

learner_err = scikitFolds(sk_trn.data, sk_trn.target, learners, 5)


print "\n" + "-" * 30
print "CROSS VALIDATION"
print "Tree 1 Accuracy: %.2f" % learner_err[0]
print "Tree 2 Accuracy: %.2f" % learner_err[1]
print "Tree 3 Accuracy: %.2f" % learner_err[2]

##################################################################################
# Create a neural network (PyBrain only)

# This is a little bit complicated, and it's also kind of slow (because it's all in Python

# The first thing we have to do is convert the data to one-of-many format.
# This sets up for one output neuron per class. If we have a single ouptut target
# with three input classifications ['A', 'B', 'C'] then we convert it to three output
# targets each with a Boolean value. This operation duplicates the original targets and
# stores them in an integer field named 'class'. I'm making a deep copy to avoid changing
# the input data for everyone else.
import copy

trn_nn = copy.deepcopy(pb_trn)
tst_nn = copy.deepcopy(pb_tst)
trn_nn._convertToOneOfMany()
tst_nn._convertToOneOfMany()

# Build a feed-forward network with 5 hidden units. We use the shortcut buildNetwork() for this.
# The input and output layer size must match the dataset's input and target dimension. You could
# add additional hidden layers by inserting more numbers giving the desired layer sizes.
#
# The output layer uses a softmax function because we are doing classification. There are more
# options to explore here, e.g. try changing the hidden layer transfer function to linear
# instead of (the default) sigmoid.
#
# Syntax for buildNetwork is buildNetwork(N_input, N_hidden, N_output, <options>)
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer

fnn = buildNetwork(trn_nn.indim, 3, trn_nn.outdim, outclass=SoftmaxLayer)

# Set up a trainer that basically takes the network and training dataset as input. For a list
# of trainers, see trainers. We are using a BackpropTrainer for this.
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError

trainer = BackpropTrainer(fnn, dataset=trn_nn, momentum=0.1, verbose=False, weightdecay=0.01)

# Train the data. You can do this in a variety of ways (like the commented-out trainUntilConvergence method)
# I'm training one epoch at a time for demonstration purposes.
epochs = 20

#trnerr,valerr = trainer.trainUntilConvergence(dataset=trn_nn,maxEpochs=epochs)
#fig_nn = plt.figure()
#ax = fig_nn.add_subplot(1, 1, 1)
#ax.plot(trnerr,'b',valerr,'r')
#plt.show()

trnerr = []
tsterr = []
for i in xrange(epochs):
    # If you set the 'verbose' trainer flag, this will print the total error as it goes.
    trainer.trainEpochs(1)
    trnresult = percentError(trainer.testOnClassData(), trn_nn['class'])
    tstresult = percentError(trainer.testOnClassData(dataset=tst_nn), tst_nn['class'])
    #print "epoch: %4d" % trainer.totalepochs, "  train error: %5.2f%%" % trnresult, "  test error: %5.2f%%" % tstresult
    trnerr.append(trnresult)
    tsterr.append(tstresult)

fig_nn = plt.figure()
ax = fig_nn.add_subplot(1, 1, 1)
ax.set_title("Neural Network Convergence")
ax.set_xlabel('Epoch')
ax.set_ylabel('Error')
ax.semilogy(range(len(trnerr)), trnerr, 'b', range(len(tsterr)), tsterr, 'r')

# Check the accuracy
print "\n" + "*" * 50
print "DEFAULT NEURAL NETWORK"
print "Training Accuracy: " + str(1 - percentError(trainer.testOnClassData(), trn_nn['class'])/100.0)
print "Testing  Accuracy: " + str(1 - percentError(trainer.testOnClassData(dataset=tst_nn), tst_nn['class'])/100.0)

# Cross Validation
# I haven't make this thing work. The above call to _convertToOneOfMany seems to conflict with the
# cross validation plumbing. There is a function
#  pybrain.tools.validation.ClassificationHelper.oneOfManyToClasses
# That does the reverse conversion. Howevever, I haven't figured out how to fit that into the cross validation
#
# Of course, we can use the scikit-learn cross validation above to do this semi-manually.
"""
from pybrain.tools.validation import CrossValidator
from pybrain.tools.validation import ModuleValidator

fnn1 = buildNetwork(trn_nn.indim, 3, trn_nn.outdim, outclass=SoftmaxLayer)
trainer1 = BackpropTrainer(fnn1, dataset=trn_nn, momentum=0.1, verbose=False, weightdecay=0.01)

fnn2 = buildNetwork(trn_nn.indim, 10, trn_nn.outdim, outclass=SoftmaxLayer)
trainer2 = BackpropTrainer(fnn, dataset=trn_nn, momentum=0.01, verbose=False, weightdecay=0.01)

fnn3 = buildNetwork(trn_nn.indim, 1, trn_nn.outdim, outclass=SoftmaxLayer)
trainer3 = BackpropTrainer(fnn, dataset=trn_nn, momentum=0.0, verbose=False, weightdecay=0.01)

validator1 = CrossValidator(trainer=trainer1, dataset=trn_nn, n_folds=5,
                            valfunc=ModuleValidator.classificationPerformance, max_epochs=20)
validator2 = CrossValidator(trainer=trainer2, dataset=trn_nn, n_folds=5,
                            valfunc=ModuleValidator.classificationPerformance, max_epochs=20)
validator3 = CrossValidator(trainer=trainer3, dataset=trn_nn, n_folds=5,
                            valfunc=ModuleValidator.classificationPerformance, max_epochs=20)

print "\n" + "-" * 30
print "CROSS VALIDATION"
print "ANN 1 Accuracy: %.2f" % validator1.validate()
print "ANN 2 Accuracy: %.2f" % validator2.validate()
print "ANN 3 Accuracy: %.2f" % validator3.validate()
"""
print "\n" + "-" * 30
print "CROSS VALIDATION"
print "  ???"
#plt.show()

##################################################################################
# Create a kNN classifier (scikit-learn. PyBrain also has one)
from  sklearn import neighbors
clf = neighbors.KNeighborsClassifier()
clf = clf.fit(sk_trn.data, sk_trn.target)

# Check the accuracy
print "\n" + "*" * 50
print "DEFAULT kNN"
print "Training Accuracy: " + str(clf.score(sk_trn.data, sk_trn.target))
print "Testing  Accuracy: " + str(clf.score(sk_tst.data, sk_tst.target))

# K-Fold validation
learners = []
learners.append(neighbors.KNeighborsClassifier())
learners.append(neighbors.KNeighborsClassifier(weights='distance', n_neighbors=30))
learners.append(neighbors.RadiusNeighborsClassifier(radius=5.0, weights='uniform'))
learner_err = scikitFolds(sk_trn.data, sk_trn.target, learners, 5)

print "\n" + "-" * 30
print "CROSS VALIDATION"
print "kNN 1 Accuracy: %.2f" % learner_err[0]
print "kNN 2 Accuracy: %.2f" % learner_err[1]
print "kNN 3 Accuracy: %.2f" % learner_err[2]

##################################################################################
# Create an SVM classifier (scikit)
# PyBrain has one but I think it might be broken (https://github.com/pybrain/pybrain/issues/104)
from sklearn import svm
clf = svm.SVC()
clf = clf.fit(sk_trn.data, sk_trn.target)

# Check the accuracy
print "\n" + "*" * 50
print "DEFAULT SVM"
print "Training Accuracy: " + str(clf.score(sk_trn.data, sk_trn.target))
print "Testing  Accuracy: " + str(clf.score(sk_tst.data, sk_tst.target))

# K-Fold validation
learners = []
learners.append(svm.SVC())
learners.append(svm.SVC(kernel='linear'))
learners.append(svm.SVC(kernel='sigmoid'))
learner_err = scikitFolds(sk_trn.data, sk_trn.target, learners, 5)

print "\n" + "-" * 30
print "CROSS VALIDATION"
print "SVM 1 Accuracy: %.2f" % learner_err[0]
print "SVM 2 Accuracy: %.2f" % learner_err[1]
print "SVM 3 Accuracy: %.2f" % learner_err[2]

##################################################################################
# Create an boosted tree classifier (scikit)
# You can find a useful example here:
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html#example-ensemble-plot-adaboost-multiclass-py
from sklearn import ensemble
clf = ensemble.AdaBoostClassifier()
clf = clf.fit(sk_trn.data, sk_trn.target)

# Check the accuracy
print "\n" + "*" * 50
print "DEFAULT BOOSTED TREE"
print "Training Accuracy: " + str(clf.score(sk_trn.data, sk_trn.target))
print "Testing  Accuracy: " + str(clf.score(sk_tst.data, sk_tst.target))

# K-Fold validation
learners = []
learners.append(ensemble.AdaBoostClassifier(n_estimators=3))
learners.append(ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2), n_estimators=1000))
learners.append(ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2), n_estimators=100))
learner_err = scikitFolds(sk_trn.data, sk_trn.target, learners, 5)

print "\n" + "-" * 30
print "CROSS VALIDATION"
print "Boosted Tree 1 Accuracy: %.2f" % learner_err[0]
print "Boosted Tree 2 Accuracy: %.2f" % learner_err[1]
print "Boosted Tree 3 Accuracy: %.2f" % learner_err[2]
