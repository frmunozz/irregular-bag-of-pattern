import pandas as pd
import numpy as np
import os

from abc import ABC, abstractmethod
import logging
import copy
import types
import matplotlib.pyplot as plt
import itertools


def get_vocabulary_size(alph_size, wl, empty_handler="special_character"):
    if empty_handler == "#":
        return (alph_size + 1) ** wl
    else:
        return alph_size ** wl


def validate_keys(kwargs, good_kwargs):
    """
    validate keyword arguments,

    checking if there is any bad keyword,
    and defining with their default value all the non-defined keywords.

    :param kwargs: the input keyword dictionary, with only the defined keywords.
    :param good_kwargs: the valid keywords with their default values.
    :return: the output keywords dictionary, with all the keywords.
    """
    good_keys = set(good_kwargs)
    bad_keys = set(kwargs) - good_keys
    if bad_keys:
        bad_keys = ", ".join(bad_keys)
        raise KeyError("Unknown parameters: {}".format(bad_keys))

    new_kwargs = {}
    for k in good_keys:
        new_kwargs[k.rstrip("_")] = kwargs.get(k, good_kwargs.get(k))
    return new_kwargs


def log_newline(self, how_many_lines=1):
    """
    Switch Logger Handler to one that generate a blank line.
    Code extracted from:
    https://stackoverflow.com/questions/20111758/how-to-insert-newline-in-python-logging/20156856

    Parameters
    ----------
    self : Object
        Logging Handler
    how_many_lines : int
        Number of lines to skip
    """

    # Switch handler, output a blank line
    self.removeHandler(self.console_handler)
    self.addHandler(self.blank_handler)
    for i in range(how_many_lines):
        self.info('')

    # Switch back
    self.removeHandler(self.blank_handler)
    self.addHandler(self.console_handler)


def create_logger(module, class_name):
    """
    Create a Logger Handler which display messages in format:

    module.ClassName.LevelMessage : textMessage

    code extracted from:
    https://stackoverflow.com/questions/20111758/how-to-insert-newline-in-python-logging/20156856

    Parameters
    ----------
    module : string
        Name of the module

    class_name : string
        Name of the class

    Returns
    -------
    logger : Logger Handler
    """
    # Create a handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt="%(name)s.%(levelname)-8s: %(message)s"))

    # Create a "blank line" handler
    blank_handler = logging.StreamHandler()
    blank_handler.setLevel(logging.DEBUG)
    blank_handler.setFormatter(logging.Formatter(fmt=''))

    # Create a logger, with the previously-defined handler
    logger = logging.getLogger("{}.{}".format(module, class_name))
    logger.propagate = False
    logger.setLevel(logging.DEBUG)  # highest level
    logger.addHandler(console_handler)

    # Save some data and add a method to logger object
    logger.console_handler = console_handler
    logger.blank_handler = blank_handler
    logger.newline = types.MethodType(log_newline, logger)

    return logger


class AbstractCore(ABC):
    _logger = None

    def __init__(self, **kwargs):
        """
        Core Class (Abstract) that define how an object manipulate variables as
        dictionary.

        This type of Class instantiation is useful when you don't know how many
        parameters the user will need and you would like to give a more wide options
        for configuring a class.

        This abstract class cannot be instantiated and must be inherited by some child
        class which need to declare all the abstract methods.

        :param kwargs:
        """
        self.kwargs = validate_keys(kwargs, copy.deepcopy(self.get_valid_kwargs()))
        super().__init__()

    def __getitem__(self, key):
        if key in self.kwargs.keys():
            return self.kwargs[key]
        else:
            return getattr(self, key)

    def __setitem__(self, key, value):
        if key in self.kwargs.keys():
            self.kwargs[key] = value
        else:
            self.logger.info("item {} does not exists in self.kwargs and cannot be set".format(key))

    @abstractmethod
    def get_valid_kwargs(self) -> dict:
        """
        Get valid keyword arguments,

        get the valid keyword arguments with their default values. This method should
        be defined in every child class.

        """
        pass

    def copy_from(self, other):
        """
        copy values from another AbstractCore,

        copy (from other class that inherit from AbstractCoreClass) all the parameters
        that match with self keyword arguments.

        :param other: other child of AbstractCoreClass
        """
        for key in self.kwargs.keys():
            if key in other.kwargs.keys():
                self.kwargs[key] = other.kwargs.get(key)

    def reset_kwargs(self):
        """
        reset the keyword arguments,

        Set their values to default.

        """
        self.kwargs = copy.deepcopy(self.get_valid_kwargs())

    @classmethod
    @abstractmethod
    def module_name(cls):
        pass

    @classmethod
    def new_logger(cls):
        """
        Class Method used to modify the class variable, in combination with self.logger(),
        we achive a singleton patter where this variable cls._logger is defined only
        once (the first time) for every instantiation of this class (is shared between
        objects.)

        """
        cls._logger = create_logger(cls.module_name(), cls.__name__)

    @property
    def logger(self) -> logging.Logger:
        """
        Get Logger Handler Using Singleton, the logger will be set only the first time
        is used, then it will use always the same Object instance. This will be shared
        between different instances of this class.

        Returns
        -------
        logger : Logger Handler
        """
        if self._logger is None:
            self.new_logger()
        return self._logger

    def blank_log(self):
        """
        create a blank line in the logger interface
        """
        self.logger.newline()


def load_pandas(path,  **kwargs):
    data_filename = kwargs.pop("data_filename")
    meta_filename = kwargs.pop("meta_filename")
    df = pd.read_csv(path + data_filename)
    df_metadata = pd.read_csv(path + meta_filename)

    passband_id = 3

    df = df[df["passband"] == passband_id]
    df = df.sort_values(by=["object_id", "mjd"])
    df_metadata = df_metadata.sort_values(by=["object_id"])
    df = df.groupby("object_id")
    fluxes = df['flux'].apply(list)
    times = df['mjd'].apply(list)
    ids = df.groups.keys()
    dataset = [np.array(fluxes.loc[i]) for i in ids]
    times_arr = []
    for i in ids:
        times_i = np.array(times.loc[i])
        times_i = times_i - times_i[0]
        times_arr.append(times_i)

    labels = df_metadata["target"].to_numpy()

    return dataset, times_arr, labels, len(dataset)


def adjust_labels(labels):
    m = len(labels)
    classes = np.sort(np.unique(labels))
    classes_count = np.zeros(len(classes), dtype=int)
    label_index = np.zeros(m)
    for i, l in enumerate(labels):
        position = np.where(classes == l)[0][0]
        classes_count[position] += 1
        label_index[i] = position

    count_sort_best_index = np.argsort(classes_count)[::-1]

    return classes_count, classes, label_index, count_sort_best_index


def sort_trim_arr(train_bop, sort_index, m, n):
    train_bop_sort = np.zeros((n+1) * m)
    idx = 0
    for j in range(n):
        k = sort_index[j]
        for i in range(m):
            train_bop_sort[idx] = train_bop[i + k * m]
            idx  += 1
    return train_bop_sort


def load_numpy_dataset(data_path, file_base):
    dataset = np.load(os.path.join(data_path, file_base % "d"), allow_pickle=True)
    times = np.load(os.path.join(data_path, file_base % "t"), allow_pickle=True)
    labels = np.load(os.path.join(data_path, file_base % "l"), allow_pickle=True)
    return dataset, times, labels, len(dataset)


def read_numpy_dataset(*args):
    dataset = np.load(args[0], allow_pickle=True)
    times = np.load(args[1], allow_pickle=True)
    labels = np.load(args[2], allow_pickle=True)
    return dataset, times, labels, len(dataset)


def read_file_regular_dataset(*args, **kwargs):
    sep = kwargs.get("sep", '\t')
    file1 = open(args[0], 'r')
    lines = file1.readlines()
    m = len(lines)
    dataset = []
    times = []
    labels = []
    for d in lines:
        arr = d[:-1].split(sep)
        y = np.array(arr[1:], dtype=float)
        t = np.arange(y.size, dtype=float)
        dataset.append(y)
        times.append(t)
        labels.append(int(arr[0]))

    return dataset, times, labels, m


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=17)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14)

    plt.ylabel('True label', fontsize=17)
    plt.xlabel('Predicted label', fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()