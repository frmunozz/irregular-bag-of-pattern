# -*- coding: utf-8 -*-
from .avocado_adapter import Featurizer


class MMMBOPFFeaturizer(Featurizer):
    """
    class used to generate the MMM-BOPF representation for the PLAsTiCC dataset.
    This class is based on AVOCADO implementation of a featurizer, but it will be adapted
    to work on MMM-BOPF method.
    """

    def extract_raw_features(self, astronomical_object, return_model=False):
        """
        extract raw features from an object

        MMM-BOPF method will generate a large amount of raw features based on
        information-retrieval theory, being the most of them zero-values. These data
        must be reduced using LSA or MANOVA techniques.


        :param astronomical_object:
        :param return_model:
        :return:
        """
        pass

    def select_features(self, raw_features):
        """
        Select features by using LSA or MANOVA methods.

        :param raw_features:
        :return:
        """
        pass