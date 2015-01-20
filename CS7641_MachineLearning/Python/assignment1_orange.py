import Orange

# Import various helpful libraries
from scipy import diag
from numpy.random import multivariate_normal, seed
import matplotlib.pyplot as plt
import subprocess

"""
Requirements
 - Orange (downloaded and compiled)
 - matplotlib
 - pydot
 - GraphViz
 - numpy
 - scipy

--------------------
Pros:
 - Covers a wide array of ML algorithms
 - Documentation (http://docs.orange.biolab.si/) covers the vast majority of the topics
 - Fast execution (written in C++)
 - Built-in function cover a whole lot of territory, especially for validation

Cons:
 - The guts are written in C++, so you can't read the algorithms. You often can't even see
 the functions.
 - Requires compilation. Not a big deal for most, but worth mentioning. PyBrain and scikit-learn
 do not require any compilation that I'm aware of.
 - It doesn't seem to be widely used, so Google searching didn't do me much good. You have to rely
 on the docs and exploring the classes dynamically.
 - You do not have access to all the knobs, only what's exposed. You will have trouble forcing a
 neural network to over-fit, for example.
 - As far as I can tell you can't track the convergence of a neural network (you just get the result)

A note on Learners and Classifiers:
Each classifier has a Learner and Classifier. Learner specifies
the algorithm (i.e. "a tree with a maximum depth of 5 and a minimum leaf population of 8". The
Classifier is an instance of the Learner, trained on specific data. Most of the Learner constructors
allow you to specify data, in which case the constructor returns a Classifier. I don't think Ensemble
learners allow you to do the skip-the-learner version.

For example

# Create a tree learner, and use it to make a tree classifier
tree_learner = Orange.classification.tree.TreeLearner(max_depth=6, min_instances=10)
tree_classifier1 = tree_learner(data)

# Skip the learner step
tree_classifier1 = Orange.classification.tree.TreeLearner(data, max_depth=6, min_instances=10)

"""

# #################################################################################
# This function computes the accuracy of a classifier. I expect there's some
# function that does this automatically, but I haven't been able to find it.
def compute_learner_accuracy(classifier, data_table):
    if not data_table:
        return None

    # I'm also not sure how to get the class (label) for a data instance. This works for this program.
    N_data = len(data_table[0])
    class_index = N_data - 1

    # Compute the number of instances and number that it get correct
    N_instances = len(data_table)
    N_correct = 0
    for entry in data_table:
        if classifier(entry) == entry[class_index]:
            N_correct += 1

    return float(N_correct) / float(N_instances)

##################################################################################
# Create a set of data points in three different classes.
# Distribute the points randomly about some XY mean with some covariance
N = 100
means = [(-1, 0), (2, 4), (3, 1)]
cov = [diag([1, 1]), diag([0.5, 1.2]), diag([1.5, 0.7])]
names = ['T1', 'T2', 'T3']

seed(25)  # For consistency
data = []
for n in xrange(N):
    for ii, name in enumerate(names):
        x, y = multivariate_normal(means[ii], cov[ii])
        data.append([x, y, name])

##################################################################################
# Create a table

# Good documentation on the table class here
# http://orange.biolab.si/docs/latest/reference/rst/Orange.data.table.html

# The table has two continuous features (X and Y) and one discrete (Type).
feature1 = Orange.feature.Continuous("X")
feature2 = Orange.feature.Continuous("Y")
classes = Orange.feature.Discrete("Type", values=names)
domain = Orange.data.Domain([feature1, feature2, classes])
table = Orange.data.Table(domain)

# Fill the table with our data
for dt in data:
    table.append(dt)

# Randomize the order
table.shuffle()

# Save the table so we can look at it.
table.save("orange_demo.tab")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# You'll notice that the Type column has a 'class' label. I'm not sure
# how Orange determines this. Perhaps it's the last column by default?
# Anyhow, that's our label.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
for entry in table:
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
# Split data into train and test
indices2 = Orange.data.sample.SubsetIndices2(p0=0.25)
ind = indices2(table)
trn_data = table.select(ind, 1)
tst_data = table.select(ind, 0)

print "\n" + "*" * 50
print "Number of data points   : " + str(len(table))
print "Number of training cases: " + str(len(trn_data))
print "Number of test cases    : " + str(len(tst_data))

##################################################################################
# Create a decision tree
tree = Orange.classification.tree.TreeLearner(trn_data)
tree.dot("orange_demo.dot")
ret = subprocess.call(['dot', '-Tgif', 'orange_demo.dot', '-oorange_demos.gif'])
if not ret == 0:
    print "Error creating the graph visualization. Please install GraphViz"

# Check the accuracy
print "\n" + "*" * 50
print "DEFAULT TREE"
print "Training Accuracy: " + str(compute_learner_accuracy(tree, trn_data))
print "Testing  Accuracy: " + str(compute_learner_accuracy(tree, tst_data))

# Cross Validation
tree1 = Orange.classification.tree.TreeLearner()
tree2 = Orange.classification.tree.TreeLearner(max_depth=3, min_instances=5)
tree3 = Orange.classification.tree.TreeLearner(max_depth=6, min_instances=10)
res = Orange.evaluation.testing.cross_validation([tree1, tree2, tree3], trn_data, folds=5)

print "\n" + "-" * 30
print "CROSS VALIDATION"
print "Tree 1 Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[0]
print "Tree 2 Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[1]
print "Tree 3 Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[2]

##################################################################################
# Create a neural network
ann = Orange.classification.neural.NeuralNetworkLearner(trn_data)

# Check the accuracy
print "\n" + "*" * 50
print "DEFAULT ANN"
print "Training Accuracy: " + str(compute_learner_accuracy(ann, trn_data))
print "Testing  Accuracy: " + str(compute_learner_accuracy(ann, tst_data))

# Cross Validation
ann1 = Orange.classification.neural.NeuralNetworkLearner()
ann2 = Orange.classification.neural.NeuralNetworkLearner(n_mid=5)
ann3 = Orange.classification.neural.NeuralNetworkLearner(n_mid=4, max_iter=3)
res = Orange.evaluation.testing.cross_validation([ann1, ann2, ann3], trn_data, folds=5)

print "\n" + "-" * 30
print "CROSS VALIDATION"
print "ANN 1 Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[0]
print "ANN 2 Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[1]
print "ANN 3 Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[2]

# Learning curve: There are several learning curve functions in Orange.evaluation.testing,
# but as far as I can tell, these aren't the "error vs iterations" curve we want in
# Neural Networks. They seem to be another form of validation, going over many iterations
# and giving the probabilistic labels for data instances. I'm not sure how to do this
# other than iterate over neural networks and rebuild each time with a different
# max_iter value.
#
# I also don't see any way to force it into over-fitting.
iterations = []
trn_error = []
tst_error = []
for ii in xrange(20):
    count = ii * 10
    ann = Orange.classification.neural.NeuralNetworkLearner(trn_data, max_iter=count)
    iterations.append(count)
    trn_error.append(1 - compute_learner_accuracy(ann, trn_data))
    tst_error.append(1 - compute_learner_accuracy(ann, tst_data))

fig_nn = plt.figure()
ax = fig_nn.add_subplot(1, 1, 1)
ax.set_title("Neural Network Convergence")
ax.set_xlabel('Iterations')
ax.set_ylabel('Error')
ax.semilogy(iterations, trn_error, 'b', iterations, tst_error, 'g')
ax.legend(['Training Error', 'Testing Error'])

##################################################################################
# Create a kNN classifier
knn = Orange.classification.knn.kNNLearner(trn_data)

# Check the accuracy
print "\n" + "*" * 50
print "DEFAULT kNN"
print "Training Accuracy: " + str(compute_learner_accuracy(knn, trn_data))
print "Testing  Accuracy: " + str(compute_learner_accuracy(knn, tst_data))

# Cross Validation
knn1 = Orange.classification.knn.kNNLearner(k=1, rank_weight=False)
knn2 = Orange.classification.knn.kNNLearner(k=5, distance_constructor=Orange.distance.Manhattan())
knn3 = Orange.classification.knn.kNNLearner(k=10, rank_weight=False)
res = Orange.evaluation.testing.cross_validation([knn1, knn2, knn3], trn_data, folds=5)

print "\n" + "-" * 30
print "CROSS VALIDATION"
print "kNN 1 Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[0]
print "kNN 2 Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[1]
print "kNN 3 Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[2]

##################################################################################
# Create an SVM classifier
svm = Orange.classification.svm.SVMLearner(trn_data)

# Check the accuracy
print "\n" + "*" * 50
print "DEFAULT SVM"
print "Training Accuracy: " + str(compute_learner_accuracy(svm, trn_data))
print "Testing  Accuracy: " + str(compute_learner_accuracy(svm, tst_data))

# Cross Validation
from Orange.classification.svm import SVMLearner, kernels
from Orange.distance import Euclidean
from Orange.distance import Hamming

svm1 = SVMLearner()

svm2 = SVMLearner()
svm2.kernel_func = kernels.RBFKernelWrapper(Hamming(trn_data), gamma=0.5)
svm2.kernel_type = SVMLearner.Custom
svm2.probability = True

svm3 = SVMLearner(kernel_type=SVMLearner.Custom,
                  kernel_func=kernels.CompositeKernelWrapper(
                      kernels.RBFKernelWrapper(Euclidean(trn_data), gamma=0.5),
                      kernels.RBFKernelWrapper(Hamming(trn_data), gamma=0.5), l=0.5),
                  probability=False)
res = Orange.evaluation.testing.cross_validation([svm1, svm2, svm3], trn_data, folds=5)

print "\n" + "-" * 30
print "CROSS VALIDATION"
print "SVM 1 Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[0]
print "SVM 2 Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[1]
print "SVM 3 Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[2]

##################################################################################
# Create an boosted Tree classifier
tree = Orange.classification.tree.TreeLearner(m_pruning=2, name="tree", max_depth=1)
boosted_learner = Orange.ensemble.boosting.BoostedLearner(tree, name="boosted tree")
boosted_classifier = boosted_learner(trn_data)

# Check the accuracy
print "\n" + "*" * 50
print "DEFAULT BOOST"
print "Training Accuracy: " + str(compute_learner_accuracy(svm, trn_data))
print "Testing  Accuracy: " + str(compute_learner_accuracy(svm, tst_data))

# Cross Validation
bst1 = Orange.ensemble.boosting.BoostedLearner(tree, name="boosted tree 1", t=1)
bst2 = Orange.ensemble.boosting.BoostedLearner(tree, name="boosted tree 2", t=5)
bst3 = Orange.ensemble.boosting.BoostedLearner(tree, name="boosted tree 3", t=10)
res = Orange.evaluation.testing.cross_validation([bst1, bst2, bst3], trn_data, folds=5)

print "\n" + "-" * 30
print "CROSS VALIDATION"
print "BST 1 Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[0]
print "BST 2 Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[1]
print "BST 3 Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[2]

# Plot the learning curve
iterations = []
trn_error = []
tst_error = []
for ii in xrange(20):
    learner = Orange.ensemble.boosting.BoostedLearner(tree, name="boosted tree 1", t=ii)
    classifier = learner(trn_data)
    iterations.append(ii)
    trn_error.append(1 - compute_learner_accuracy(classifier, trn_data))
    tst_error.append(1 - compute_learner_accuracy(classifier, tst_data))

fig_nn = plt.figure()
ax = fig_nn.add_subplot(1, 1, 1)
ax.set_title("Boosted Tree Convergence")
ax.set_xlabel('Iterations')
ax.set_ylabel('Error')
ax.semilogy(iterations, trn_error, 'b', iterations, tst_error, 'g')
ax.legend(['Training Error', 'Testing Error'])

plt.savefig("orange_demo_boosted_tree_error.png")
