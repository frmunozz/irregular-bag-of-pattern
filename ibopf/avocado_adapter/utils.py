import os
from ..settings import settings, get_path, get_data_directory


def get_classifier_path(name, method="IBOPF"):
    """Get the path to where a classifier should be stored on disk

    Parameters
    ----------
    name : str
        The unique name for the classifier.
    """
    # classifier_directory = settings[method]["classifier_directory"]
    classifier_directory = get_path(method, "classifier_directory")
    classifier_path = os.path.join(classifier_directory, "classifier_%s.pkl" % name)

    return classifier_path