import avocado
import numpy as np
from scipy.special import erf
from sklearn.pipeline import Pipeline


class PlasticcAugmentor(avocado.plasticc.PlasticcAugmentor):

    def __init__(self):
        super(PlasticcAugmentor, self).__init__()
        self._min_detections = 2

    def augment_object(self, reference_object, force_success=True, custom_detections=True):
        if custom_detections:
            self._min_detections = np.sum(reference_object.observations["detected"])
        aug = super(PlasticcAugmentor, self).augment_object(reference_object, force_success=force_success)
        self._min_detections = 2
        return aug

    def _simulate_detection(self, observations, augmented_metadata):
        """Simulate the detection process for a light curve.
        We model the PLAsTiCC detection probabilities with an error function.
        I'm not entirely sure why this isn't deterministic. The full light
        curve is considered to be detected if there are at least 2 individual
        detected observations.
        Parameters
        ==========
        observations : pandas.DataFrame
            The augmented observations that have been sampled from a Gaussian
            Process.
        augmented_metadata : dict
            The augmented metadata
        Returns
        =======
        observations : pandas.DataFrame
            The observations with the detected flag set.
        pass_detection : bool
            Whether or not the full light curve passes the detection thresholds
            used for the full sample.
        """
        s2n = np.abs(observations["flux"]) / observations["flux_error"]
        prob_detected = (erf((s2n - 5.5) / 2) + 1) / 2.0
        observations["detected"] = np.random.rand(len(s2n)) < prob_detected

        pass_detection = np.sum(observations["detected"]) >= self._min_detections

        return observations, pass_detection


class PlasticcFeaturizer(avocado.plasticc.PlasticcFeaturizer):

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def extract_raw_features(self, astronomical_object, return_model=False):
        pass

    def select_features(self, raw_features):
        pass


class Dataset(avocado.Dataset):
    pass


class AstronomicalObject(avocado.AstronomicalObject):
    pass