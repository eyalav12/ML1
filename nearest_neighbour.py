import numpy as np
import matplotlib.pyplot as plt
import numpy.random


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'
    # print("len x and m:")
    # print(len(x_list))
    # print(m)
    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    return k, x_train, y_train


def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    k, x_train, y_train = classifier

    def classify_example(example):
        def get_distance(sample):
            return np.linalg.norm(example - sample)

        distance_array = np.zeros(len(x_train))
        for i in range(len(x_train)):
            distance_array[i] = get_distance(x_train[i])

        # indices of the k-smallest values
        k_nearest_neighbors = np.argpartition(distance_array, k)[: k]

        # turn index into appropriate label
        for i in range(k):
            k_nearest_neighbors[i] = y_train[k_nearest_neighbors[i]]

        num_of_occurrences = np.bincount(k_nearest_neighbors)

        return np.argmax(num_of_occurrences)

    y_test = np.zeros(len(x_test))

    for i in range(len(y_test)):
        y_test[i] = classify_example(x_test[i])

    return y_test.reshape(len(y_test), 1)


def example_input():
    """
    Example for input from page 3 of the exercise file.
    Expected to return [1,0,0,1]
    """
    k = 1
    x_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 0, 1])
    classifier = learnknn(k, x_train, y_train)
    x_test = np.array([[10, 11], [3.1, 4.2], [2.9, 4.2], [5, 6]])
    y_test_prediction = predictknn(classifier, x_test)
    print(y_test_prediction)


def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifier = learnknn(5, x_train, y_train)

    preds = predictknn(classifier, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


def question2_a():
    data = np.load('mnist_all.npz')

    train2 = data['train2']
    train3 = data['train3']
    train5 = data['train5']
    train6 = data['train6']

    test2 = data['test2']
    test3 = data['test3']
    test5 = data['test5']
    test6 = data['test6']

    total_test_examples = len(test2) + len(test3) + len(test5) + len(test6)

    # sample sizes
    x_axis = []

    y_axis_mean_error = []
    y_axis_max_error = []
    y_axis_min_error = []

    for sample_size in range(5, 101, 5):
        x_axis.append(sample_size)

        mean_errors = []
        max_error = 0
        min_error = 1

        for i in range(10):
            x_train, y_train = gensmallm([train2, train3, train5, train6], [2, 3, 5, 6], sample_size)

            x_test, y_test = gensmallm([test2, test3, test5, test6], [2, 3, 5, 6], total_test_examples)

            k = 1
            classifier = learnknn(k, x_train, y_train)
            predictions = predictknn(classifier, x_test)

            y_test = y_test.reshape(len(y_test), 1)

            current_error = np.mean(y_test != predictions)
            max_error = max(max_error, current_error)
            min_error = min(min_error, current_error)

            mean_errors.append(current_error)

        average_mean_error = sum(mean_errors) / len(mean_errors)

        y_axis_mean_error.append(average_mean_error)
        y_axis_max_error.append(max_error - average_mean_error)
        y_axis_min_error.append(average_mean_error - min_error)

    plt.plot(x_axis, y_axis_mean_error, "k", label="mean error")
    min_and_max_errors = [y_axis_min_error, y_axis_max_error]
    plt.errorbar(x_axis, y_axis_mean_error, min_and_max_errors, fmt='o', ecolor='red')

    plt.title("Nearest Neighbor Error")
    plt.xlabel("sample size")
    plt.ylabel("Mean error")

    plt.show()


def question2_e():
    data = np.load('mnist_all.npz')

    train2 = data['train2']
    train3 = data['train3']
    train5 = data['train5']
    train6 = data['train6']

    test2 = data['test2']
    test3 = data['test3']
    test5 = data['test5']
    test6 = data['test6']

    sample_size = 200
    total_test_examples = len(test2) + len(test3) + len(test5) + len(test6)

    x_train, y_train = gensmallm([train2, train3, train5, train6], [2, 3, 5, 6], sample_size)
    x_test, y_test = gensmallm([test2, test3, test5, test6], [2, 3, 5, 6], total_test_examples)

    # number of nearest neighbors (k)
    x_axis = []

    # average mean errors
    y_axis = []

    for k in range(1, 12):
        x_axis.append(k)
        mean_errors = []

        for i in range(10):
            classifier = learnknn(k, x_train, y_train)
            predictions = predictknn(classifier, x_test)

            y_test = y_test.reshape(len(y_test), 1)

            error = np.mean(y_test != predictions)
            mean_errors.append(error)
            print(error)

        average_mean_error = sum(mean_errors) / len(mean_errors)
        y_axis.append(average_mean_error)

    plt.plot(x_axis, y_axis)

    plt.title("K-nn average mean error")
    plt.xlabel("K")
    plt.ylabel("Average mean error")

    plt.show()


def question2_f():
    data = np.load('mnist_all.npz')

    train2 = data['train2']
    train3 = data['train3']
    train5 = data['train5']
    train6 = data['train6']

    total_sample_size = 200
    uncorrupted_sample_size = int(0.85 * total_sample_size)
    corrupted_sample_size = int(0.15 * total_sample_size)

    x_train_uncorrupted, y_train_uncorrupted = gensmallm([train2, train3, train5, train6], [2, 3, 5, 6],
                                                         uncorrupted_sample_size)

    x_train_corrupted, y_train_corrupted = gensmallm([train2, train3, train5, train6],
                                                     [numpy.random.choice([3, 5, 6]), numpy.random.choice([2, 5, 6]),
                                                      numpy.random.choice([2, 3, 6]), numpy.random.choice([3, 5, 6])],
                                                     corrupted_sample_size)

    test2 = data['test2']
    test3 = data['test3']
    test5 = data['test5']
    test6 = data['test6']

    total_test_examples = len(test2) + len(test3) + len(test5) + len(test6)
    uncorrupted_test_examples_size = int(0.85 * total_test_examples)
    corrupted_test_examples_size = int(0.15 * total_test_examples)

    x_test_uncorrupted, y_test_uncorrupted = gensmallm([test2, test3, test5, test6], [2, 3, 5, 6],
                                                       uncorrupted_test_examples_size)

    x_test_corrupted, y_test_corrupted = gensmallm([test2, test3, test5, test6],
                                                   [numpy.random.choice([3, 5, 6]), numpy.random.choice([2, 5, 6]),
                                                    numpy.random.choice([2, 3, 6]), numpy.random.choice([3, 5, 6])],
                                                   corrupted_test_examples_size)

    x_train, y_train = np.concatenate((x_train_uncorrupted, x_train_corrupted), axis=0), np.concatenate(
        (y_train_uncorrupted,
         y_train_corrupted), axis=0)

    x_test, y_test = np.concatenate((x_test_uncorrupted, x_test_corrupted), axis=0), np.concatenate(
        (y_test_uncorrupted, y_test_corrupted), axis=0)

    # same code of question 2e
    # number of nearest neighbors (k)
    x_axis = []

    # average mean errors
    y_axis = []
    for k in range(1, 12):
        x_axis.append(k)
        mean_errors = []

        for i in range(10):
            classifier = learnknn(k, x_train, y_train)
            predictions = predictknn(classifier, x_test)

            y_test = y_test.reshape(len(y_test), 1)

            error = np.mean(y_test != predictions)
            mean_errors.append(error)

        average_mean_error = sum(mean_errors) / len(mean_errors)
        y_axis.append(average_mean_error)

    plt.plot(x_axis, y_axis)

    plt.title("K-nn average mean error - Corrupted case")
    plt.xlabel("K")
    plt.ylabel("Average mean error")

    plt.show()


if __name__ == '__main__':
    simple_test()
