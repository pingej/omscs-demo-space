import sys
import os
import time
import random

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import shared.ConvergenceTrainer as ConvergenceTrainer
import shared.DataSet as DataSet
import shared.Instance as Instance
import shared.SumOfSquaresError as SumOfSquaresError
import func.nn.backprop as BackProp

class ConfusionMatrix:
    def __init__(self, class_names):
        self.index = {}
        self.matrix = []
        for cc, name in enumerate(class_names):
            self.matrix.append([0 for cls in class_names])
            self.index[name] = cc

    def add_by_index(self, true_index, computed_index):
        self.matrix[true_index][computed_index] += 1

    def add_by_name(self, true_name, computed_name):
        self.add_by_index(self.index[true_name], self.index[computed_name])

    def total_entries(self):
        n_classes = len(self.matrix)

        total = 0
        for tt in xrange(n_classes):
            for oo in xrange(n_classes):
                total += self.matrix[tt][oo]
        return total

    def accurate_predictions(self):
        n_classes = len(self.matrix)
        accurate_predictions = 0
        for tt in xrange(n_classes):
            accurate_predictions += self.matrix[tt][tt]
        return accurate_predictions

    def pct_accuracy(self):
        n_correct = self.accurate_predictions()
        n_total = self.total_entries()
        return float(n_correct)/float(n_total)

    def write(self, filename):
        fout = open(filename, 'w')
        n_classes = len(self.matrix)
        for tt in xrange(n_classes):
            for oo in xrange(n_classes):
                fout.write(str(self.matrix[tt][oo]) + ",")
            fout.write("\n")
        fout.close()

    def display(self):
        print("Accuracy: " + str(self.pct_accuracy()))

class BinaryConfusionMatrix(ConfusionMatrix):

    def __init__(self):
        # We're forcing the negative class to 0 and the positive class to 1
        super(BinaryConfusionMatrix, self).__init__(["False", "True"])

    def num_true_positives(self):
        return self.matrix[1][1]

    def num_false_positives(self):
        return self.matrix[0][1]

    def num_true_negatives(self):
        return self.matrix[0][0]

    def num_false_negatives(self):
        return self.matrix[1][0]

    def pct_false_positives(self):
        return self.num_false_negatives()/float(self.total_entries)

    def pct_false_negatives(self):
        return self.num_false_negatives()/float(self.total_entries)

    def precision(self):
        # Look at http://en.wikipedia.org/wiki/Precision_and_recall
        # Precision is <true positives>/<total positives + false positives>
        return self.num_true_positives()/float(self.num_true_positives() + self.num_false_positives())

    def recall(self):
        # Look at http://en.wikipedia.org/wiki/Precision_and_recall
        # Recall is <true positives>/<total positives + false negatives>
        return self.num_true_positives()/float(self.num_true_positives() + self.num_false_negatives())

    def display(self):
        super(BinaryConfusionMatrix, self).display()
        print("Precision: " + str(self.precision()))
        print("Recall: " + str(self.recall()))

class ScikitLearnNeuralNetwork:

    def __init__(self):
        # Basic input data
        self.verbose = False
        self.dataset_name = ""
        self.n_samples = 0
        self.n_features = 0
        self.n_targets = 0
        self.samples = []

        # Output types may be binary or multiclass
        # If binary, we must use an output threshold to decide if an instance is in or out.
        self.out_type = "binary"
        self.single_class_threshold = 0.5

        # The number of inputs and outputs are fixed by your features and targets. However you can
        # configure any number of hidden layers. This list is
        self.middle_layers = []

        # I'm making this instance data to save copying and memory.
        self.training_data = []
        self.testing_data = []
        self.neural_network = None

    def read_data_files(self):
        """
        Read a scikit data set. The entire file should be numbers, so I'm not bothering with CSV or anything fancy.
        Just numbers separated by spaces.
        :return: None
        """

        # Set the data file names
        data_file = self.dataset_name + ".data"
        target_file = self.dataset_name + ".target"

        # Check that the files exist
        if not os.path.isfile(data_file):
            raise Exception("Data file '" + data_file + "' not found")
        if not os.path.isfile(data_file):
            raise Exception("Target file '" + target_file + "' not found")

        # Read the lines of the data and target files
        if self.verbose:
            print("Loading data")
        d_in = open(data_file, 'r')
        data_lines = d_in.readlines()
        d_in.close()

        t_in = open(target_file, 'r')
        target_lines = t_in.readlines()
        t_in.close()

        # A quick check that there is a one-to-one correspondence between data and target lines
        self.n_samples = len(data_lines)
        if not self.n_samples == len(target_lines):
            raise Exception("Data and Target lengths are not the same.")

        # Interpret each data and target line pair
        if self.verbose:
            print("Interpreting data")
        self.samples = []
        self.n_features = len(data_lines[0].split())
        self.n_targets = len(target_lines[0].split())
        for ss in xrange(self.n_samples):

            data = []
            ds = data_lines[ss].split()
            n_inputs = len(ds)
            for val in ds:
                data.append(float(val))

            target = []
            ds = target_lines[ss].split()
            n_outputs = len(ds)
            for val in ds:
                target.append(float(val))
            inst = Instance(data)
            inst.setLabel(Instance(target))

            # Do some checking before we append this
            if not self.n_features == n_inputs:
                raise Exception("Line " + str(ss) + ": Number of data points does not match previous lines")
            if not self.n_targets == n_outputs:
                raise Exception("Line " + str(ss) + ": Number of targets does not match previous lines")

            # Append this data pattern
            self.samples.append(inst)

        # What type of classification is this?
        if self.n_targets == 1:
            self.out_type = "binary"
        else:
            self.out_type = "multiclass"


    def subsample_instances(self, instances, indices):
        """
        Return the instances at the specified indices
        :param instances: Data samples
        :param indices: Indices of the samples to keep
        :return: Subsampled instances
        """
        return [instances[index] for index in indices]


    def subsample_instances_fixed_number(self, instances, number_of_samples):
        """
        Extracts a fixed number of samples without replacement
        :param instances: Data samples
        :param number_of_samples: The number of samples to extract
        :return: Subsampled instances
        """
        indices = random.sample(len(instances), number_of_samples)
        return self.subsample_instances(instances, indices)


    def split_data(self, test_proportion, max_entries=None):
        """
        Split data into two pieces in the stated proportion
        :param instances: Data samples
        :param test_proportion: Split between the two sections
        :param max_entries: Maximum number of entries to keep total
        :return: (training_samples, test_samples)
        """
        indices = range(self.n_samples)
        random.shuffle(indices)

        Ndata = self.n_samples
        if max_entries:
            Ndata = max_entries
            indices = indices[:max_entries]

        # Split the data
        separator = int(Ndata * test_proportion)
        test_indices = indices[:separator]
        train_indices = indices[separator:]
        self.training_data = self.subsample_instances(self.samples, train_indices)
        self.testing_data = self.subsample_instances(self.samples, test_indices)

    def get_node_count_list(self):
        node_counts = [self.n_features]
        for mm in self.middle_layers:
            node_counts.append(mm)
        node_counts.append(self.n_targets)
        return node_counts

    def train_backpropagation(self):
        """
        Train a neural network on this topology
        :param instances: Instances used to train the network
        :param middle_topology: A list of the middle layer node counts (may be empty)
        :return: None
        """
        dataset = DataSet(self.training_data)

        ####################################################
        # Build a neural network
        factory = BackProp.BackPropagationNetworkFactory()
        self.neural_network = factory.createClassificationNetwork(self.get_node_count_list())

        ####################################################
        # Train the network
        print("Training network")
        bpct = BackProp.BatchBackPropagationTrainer(dataset, self.neural_network, SumOfSquaresError(), BackProp.RPROPUpdateRule())
        trainer = ConvergenceTrainer(bpct)
        trainer.train()

    def _get_sample_output_as_list(self, sample):
        # Get the true class as a list
        true_output_instance = sample.getLabel()
        true_output = []
        for tt in xrange(self.n_targets):
            true_output.append(true_output_instance.getContinuous(tt))
        return true_output

    def _get_current_computed_output_as_list(self):
        computed_output_vector = self.neural_network.getOutputValues()
        computed_output = []
        for tt in xrange(self.n_targets):
            computed_output.append(computed_output_vector.get(tt))
        return computed_output

    def _check_accuracy_binary(self, data):
        if not self.n_targets == 1:
            raise Exception("Binary classification only allowed for single-target outputs")

        conf = BinaryConfusionMatrix()

        for sample in data:
            self.neural_network.setInputValues(sample.getData())
            self.neural_network.run()

            true_output = self._get_sample_output_as_list(sample)
            computed_output = self._get_current_computed_output_as_list()

            # The classes are chosen by being greater than the threshold
            true_class = 0
            if true_output[0] > self.single_class_threshold:
                true_class = 1
            computed_class = 0
            if computed_output[0] > self.single_class_threshold:
                computed_class = 1

            conf.add_by_index(true_class, computed_class)

        # A quick error check
        if not conf.total_entries() == len(data):
            raise Exception("Error computing binary confusion matrix: Confusion matrix and"
                            "data should have the same number of entries")

        # Return the confusion matrix
        return conf

    def _check_accuracy_multiclass(self, data):
        if not self.n_targets > 1:
            raise Exception("Multiclass classification only allowed for multi-target outputs")

        conf = ConfusionMatrix(range(self.n_targets))
        for sample in data:
            self.neural_network.setInputValues(sample.getData())
            self.neural_network.run()

            true_output = self._get_sample_output_as_list(sample)
            computed_output = self._get_current_computed_output_as_list()

            true_class = true_output.index(max(true_output))
            computed_class = computed_output.index(max(computed_output))
            conf.add_by_index(true_class, computed_class)

        # A quick error check
        if not conf.total_entries() == len(data):
            raise Exception("Error computing binary confusion matrix: Confusion matrix and"
                            "data should have the same number of entries")

        # Return the confusion matrix
        return conf

    def check_accuracy(self, data):
        if self.out_type == "binary":
            return self._check_accuracy_binary(data)
        elif self.out_type == "multiclass":
            return self._check_accuracy_multiclass(data)
        else:
            raise Exception("Unknown output type '" + self.out_type + "'")

    def basic_run(self, split_proportion=0.25):
        # Load the data set
        self.read_data_files()

        # Split the test and training data
        self.split_data(split_proportion)

        # Train the networks
        self.train_backpropagation()

        # Check the output
        training_confusion = self.check_accuracy(self.training_data)
        testing_confusion = self.check_accuracy(self.testing_data)

        # Save the confusion matrices
        training_confusion.write("confusion_training.txt")
        testing_confusion.write("confusion_testing.txt")

        # Check basic accuracy
        print("\n\nTraining Results")
        training_confusion.display()
        print("\n\nTesting Results")
        testing_confusion.display()

def iris_demo():
    # Load and execute the iris example
    nn = ScikitLearnNeuralNetwork()
    nn.dataset_name = "iris/iris"

    nn.basic_run()


if __name__ == "__main__":
    iris_demo()