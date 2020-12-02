from collections import defaultdict
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from scipy.stats import norm, linregress
import matplotlib.pyplot as plt


def mean_value_bp(values, alphabet_size, strategy="uniform"):
	if strategy == "uniform":
		values_min = np.min(values)
		values_max = np.max(values)
		return np.linspace(values_min, values_max, alphabet_size+1)[1:-1]
	elif strategy == "normal":
		return norm.ppf(np.linspace(0, 1, alphabet_size+1)[1:-1], np.mean(values), np.std(values))

		
def slope_bp(alphabet_size):
	values_min = -np.pi/4
	values_max = np.pi/4
	return np.linspace(values_min, values_max, alphabet_size+1)[1:-1]


class BagOfPattern(object):
	def __init__(self, alphabet_size_unit, feature_length, feature_type="trend_value",
		global_break_points=False, bp_strategy="uniform"):
		self.feature_type = feature_type
		self.feature_length = feature_length
		self.alph_unit = alphabet_size_unit
		self.alph_size = self._get_alph_size(alphabet_size_unit)
		self.bop_size = self.get_bop_size()
		self.global_break_points = global_break_points
		self.bp_strategy = bp_strategy

	def get_bop_size(self):
		return self.alph_size ** self.feature_length

	def _get_alph_size(self, alph_size_unit):
		if self.feature_type == "trend_value":
			return alph_size_unit * alph_size_unit + 1
		elif self.feature_type == "mean":
			return alph_size_unit


	def _mean_value_char(self, i, j, break_points):
		mean = np.mean(self.observations[i:j])
		character_idx = np.digitize(mean, break_points)
		return character_idx

	def _trend_value_char(self, i, j, break_points):
		value = np.mean(self.observations[i:j])
		slope, _,_,_,_ = linregress(self.time_stamp[i:j], self.observations[i:j])
		trend = np.arctan(slope)
		value_char_idx = np.digitize(value, break_points[0])
		slope_char_idx = np.digitize(slope, break_points[1])
		return slope_char_idx, value_char_idx

	def get_break_points(self, observations, i, j):
		if self.feature_type == "trend_value":
			if self.global_break_points:
				v_bp = mean_value_bp(observations, self.alph_unit, strategy=self.bp_strategy)
			else:
				v_bp = mean_value_bp(observations[i:j], self.alph_unit)
			s_bp = slope_bp(self.alph_unit)
			return [s_bp, v_bp]
		elif self.feature_type == "mean":
			if self.global_break_points:
				return mean_value_bp(observations, self.alph_unit, strategy=self.bp_strategy)
			else:
				return mean_value_bp(observations[i:j], self.alph_unit)

	def get_min_limit(self, tol):
		return self.feature_length * tol

	def segment_to_char(self, obs, times, break_points):
		if self.feature_type == "trend_value":
			mean = np.mean(obs)
			slope, _,_,_,_ = linregress(obs, times)
			trend = np.arctan(slope)
			value_char_idx = np.digitize(mean, break_points[0])
			trend_char_idx = np.digitize(trend, break_points[1])
			return value_char_idx + self.alph_unit * trend_char_idx

		elif self.feature_type == "mean":
			mean = np.mean(obs)
			return np.digitize(mean, break_points)

	def sequence_to_word(self, i, j, ini_time, end_time, observations, time_stamp):
		char_windows = np.linspace(ini_time, end_time, 
					self.feature_length+1)[1:-1]
		break_points = self.get_break_points(observations, i, j)

		sub_obs = observations[i:j]
		sub_times = time_stamp[i:j]

		segments = np.digitize(sub_times, char_windows)

		word_idx = 0
		for k in range(self.feature_length):
			idxs = np.where(k == segments)[0]
			if len(idxs) > 1:
				sub_sub_obs = sub_obs[idxs]
				sub_sub_times = sub_times[idxs]
				val = self.segment_to_char(sub_sub_obs, sub_sub_times, break_points)
			else:
				val = self.alph_size - 1

			word_idx += (self.alph_size ** k) * val
		return word_idx


	def transform_fixed_step(self, observations, time_stamp, window, time_step, tol = 2):
		i = 0
		j = 1
		n = observations.size
		ini_time = time_stamp[0]
		end_time = ini_time + window
		bop = np.zeros(self.bop_size)
		pword_idx = -1
		while ini_time < time_stamp[-1]:
			while time_stamp[i] < ini_time:
				i += 1
			while time_stamp[j] <= end_time:
				if j == n-1:
					break
				j += 1

			if j - i > self.get_min_limit(tol):
				wordp_idx = self.sequence_to_word(i, j, ini_time, end_time,
					observations, time_stamp)
				if wordp_idx != pword_idx:
					bop[wordp_idx] += 1
					pword_idx = wordp_idx
			# else:
				# print("range: ", i, j, ini_time, end_time, "cannot be transformed")
			ini_time += time_step
			end_time += time_step

		return bop



class ExtendedBOPBase(object):
	def __init__(self, alph_unit, word_length, feature_type="mean", global_break_points=False,
		bp_strategy="uniform", tol=2):
		if isinstance(word_length, int):
			word_length = np.array([word_length])
		elif isinstance(word_length, list):
			word_length = np.array(word_length)

		self.alph_unit = alph_unit
		self.feature_type = feature_type
		self.global_break_points = global_break_points
		self.bp_strategy = bp_strategy
		self.word_length = word_length
		self.tol = tol
		self.alph_size = self.get_alph_size()
		self.bop_size = self.get_bop_size()
		self.pword = -1

	def get_alph_size(self):
		if self.feature_type == "trend_value":
			return self.alph_unit * self.alph_unit + 1
		elif self.feature_type == "mean":
			return self.alph_unit + 1

	def get_bop_size(self):
		return np.sum(self.alph_size ** self.word_length)

	def get_break_points(self, observations):
		if self.feature_type == "mean":
			v_bp = mean_value_bp(observations, self.alph_unit, strategy=self.bp_strategy)
			return [v_bp]
		elif self.feature_type == "trend_value":
			v_bp = mean_value_bp(observations, self.alph_unit, strategy=self.bp_strategy)
			s_bp = slope_bp(self.alph_unit)
			return [v_bp, s_bp]

	def segment_to_char(observations, time_stamps, i, j, break_points):
		if self.feature_type == "mean":
			mean = np.mean(observations[i:j])
			return np.digitize(mean, break_points[0])
		elif self.feature_type == "trend_value":
			mean = np.mean(observations[i:j])
			slope, _,_,_,_ = linregress(observations[i:j], time_stamps[i:j])
			trend = np.arctan(slope)
			idx1 = np.digitize(mean, break_points[0])
			idx2 = np.digitize(trend, break_points[1])
			return idx1 + self.alph_unit * idx2
		else:
			raise ValueError("feature type '%s' unknown" % self.feature_type)


	def sequence_to_word(self, observations, time_stamps, i, j, ini_time, end_time, break_points):
		bop_offset = 0
		words_idxs = []
		words_valid = []
		for i in range(len(self.word_length)):
			wl = self.word_length[i]
			char_invalids = 0
			if wl > 1:
				seg_limits = np.linspace(ini_time, end_time, wl + 1)[1:-1]
				segments = np.digitize(time_stamps[i:j], seg_limits)

				wordp = 0
				char_ini = 0
				for k in range(wl):
					char_end = np.argmax(segments[char_ini:] > k)
					if char_end > self.tol:
						char_end += char_ini
						val = self.segment_to_char(observations, time_stamps, char_ini, char_end, break_points)
					else:
						char_invalids += 1
						val = self.alph_size - 1
					wordp += (self.alph_size ** k) * val

			else:
				if j - i > self.tol:
					char = self.segment_to_char(observations, time_stamps, i, j, break_points)
				else:
					char_invalids += 1
					char = self.alph_size - 1
				wordp = char
			words_valid.append(char_invalids <= min(wl // 2, 2))
			words_idxs.append(bop_offset + wordp)
			bop_offset += self.alph_size ** wl
		return words_idxs, words_valid

	def segment_to_char(self, observations, time_stamps, i, j, break_points):
		if self.feature_type == "mean":
			mean = np.mean(observations[i:j])
			return np.digitize(mean, break_points[0])
		elif self.feature_type == "trend_value":
			mean = np.mean(observations[i:j])
			slope, _,_,_,_ = linregress(observations[i:j], time_stamps[i:j])
			trend = np.arctan(slope)
			idx1 = np.digitize(mean, break_points[0])
			idx2 = np.digitize(trend, break_points[1])
			return idx1 + self.alph_unit * idx2
		else:
			raise ValueError("feature type '%s' unknown" % self.feature_type)


class MultiWordBOP(ExtendedBOPBase):

	def transform_fixed_step(self, observations, time_stamps, window, window_step):
		bop = np.zeros(self.bop_size)
		i = 0
		j = 1
		n = observations.size
		min_tol = np.min(self.word_length) * self.tol
		ini_obs = time_stamps[0]
		end_obs = time_stamps[n-1]
		n_segments = int(np.ceil(((end_obs - ini_obs) - window) / window_step))
		break_points = None
		if self.global_break_points:
			break_points = self.get_break_points(observations)
		for k in range(n_segments):
			ini_time = ini_obs + k*window_step
			end_time = ini_time + window
			i2 = np.argmax(time_stamps[i:] >= ini_time)
			j2 = np.argmax(time_stamps[j:] > end_time)
			i += i2
			j += j2
			if (i2 != 0 or j2 != 0) and j - i > min_tol:
				if break_points is None:
					break_points = self.get_break_points(observations[i:j])
				word_idxs = self.sequence_to_word(observations, time_stamps, i, j, ini_time, end_time, break_points)
				for word_idx in word_idxs:
					bop[word_idx] += 1
		return bop

class MultiWindowBOP(ExtendedBOPBase):

	def transform_fixed_step(self, observations, time_stamps, windows, windows_step):
		bop = np.zeros(self.bop_size * len(windows))
		offset = 0
		for window, window_step in zip(windows, windows_step):
			i = 0
			j = 1
			n = observations.size
			ini_obs = time_stamps[0]
			end_obs = time_stamps[n-1]
			n_segments = int(np.ceil((end_obs - ini_obs) - window) / window_step)
			break_points = None
			if self.global_break_points:
				break_points = self.get_break_points(observations)
			for k in range(n_segments):
				ini_time = ini_obs + k*window_step
				end_time = ini_time + window
				i2 = np.argmax(time_stamps[i:] >= ini_time)
				j2 = np.argmax(time_stamps[j:] > end_time)
				i += i2
				j += j2
				if (i2 != 0 or j2 != 0) and j - i > self.tol * self.word_length[0]:
					if break_points is None:
						break_points = self.get_break_points(observations[i:j])
					word_idx = self.sequence_to_word(observations, time_stamps, i, j, ini_time, end_time, break_points)
					bop[word_idx[0] + offset] += 1
			offset += self.bop_size
		return bop

	def transform_all_subsequence(self, observations, time_stamps, windows):
		bop = np.zeros(self.bop_size * len(windows))
		pword = [-1] * len(self.word_length)
		offset = 0
		for window, window_step in zip(windows, windows_step):
			i = 0
			j = 1
			n = observations.size
			ini_obs = time_stamps[0]
			end_obs = time_stamps[n-1]
			break_points = None
			if self.global_break_points:
				break_points = self.get_break_points(observations)
			while i < n - self.tol and time_stamps[i] < end_obs:
				ini_time = time_stamps[i]
				end_time = ini_time + window
				while time_stamps[j] < end_time:
					if j == n-1:
						break
					j += 1
				if j - i > self.tol * self.word_length[0]:
					if not self.global_break_points:
						break_points = self.get_break_points(observations[i:j])
					word_idx, words_valid = self.sequence_to_word(observations, time_stamps, i, j, ini_time, end_time, break_points)
					for k in range(len(word_idx)):
						if words_valid[k]:
							if pword[k] != word_idx[k]:
								bop[word_idx[k] + offset] += 1
								pword[k] = word_idx[k]
				i += 1
			offset += self.bop_size
		return bop


class MultiFeatureBOP(object):
	def __init__(self, alph_unit, word_length, features=["mean"], global_break_points=False,
		bp_strategy="uniform", tol=2):
		if isinstance(word_length, int):
			word_length = np.array([word_length])
		elif isinstance(word_length, list):
			word_length = np.array(word_length)

		self.alph_unit = alph_unit
		self.features = features
		self.global_break_points = global_break_points
		self.bp_strategy = bp_strategy
		self.word_length = word_length
		self.tol = tol
		self.alph_size = self.get_alph_size()
		self.bop_size = self.get_bop_size()
		self.pword = -1

	def get_alph_size(self):
		return (self.alph_unit + 1) * len(self.features)

	def get_bop_size(self):
		return np.sum(self.alph_size ** self.word_length) * len(self.features)

	def get_break_points(self, observations):
		bps = []
		for fea in self.features:
			if self.feature_type == "mean":
				bp = mean_value_bp(observations, self.alph_unit, strategy=self.bp_strategy)
			elif self.feature_type == "typerend":
				bp = slope_bp(self.alph_unit)
			else:
				raise ValueError("feature type '%s' unknown" % fea)
			bps.append(bp)
		return bps

	def segment_to_char(observations, time_stamps, i, j, break_points):
		idxs = []
		for i, fea in enumerate(self.features):
			if fea == "mean":
				mean = np.mean(observations[i:j])
				idxs.append(i*(self.alph_unit+1) + np.digitize(mean, break_points[i]))
			elif fea == "trend":
				slope, _,_,_,_ = linregress(observations[i:j], time_stamps[i:j])
				trend = np.arctan(slope)
				idxs.append(i*(self.alph_unit+1) + np.digitize(trend, break_points[i]))
			else:
				raise ValueError("feature type '%s' unknown" % fea)

		return idxs

	def sequence_to_word(self, observations, time_stamps, i, j, ini_time, end_time, break_points):
		bop_offset = 0
		words_idxs = []
		words_valid = []
		for i in range(len(self.word_length)):
			wl = self.word_length[i]
			char_invalids = 0
			wordp = np.zeros(len(self.features))
			if wl > 1:
				seg_limits = np.linspace(ini_time, end_time, wl + 1)[1:-1]
				segments = np.digitize(time_stamps[i:j], seg_limits)
			
				char_ini = 0
				for k in range(wl):
					char_end = np.argmax(segments[char_ini:] > k)
					if char_end > self.tol:
						char_end += char_ini
						vals = self.segment_to_char(observations, time_stamps, char_ini, char_end, break_points)
					else:
						char_invalids += 1
						vals = [(i+1)*self.alph_size -1 for i in range(len(self.features))]
					wordp += (self.alph_size ** k) * vals
			else:
				if j - i > self.tol:
					vals = self.segment_to_char(observations, time_stamps, i, j, break_points)
				else:
					char_invalids += 1
					vals = [(i+1)*self.alph_size -1 for i in range(len(self.features))]
				wordp = vals
			for wp in wordp:
				words_valid.append(char_invalids <= wd // 2 if wd > 2 else 0)
				words_idxs.append(bop_offset + wp)
			bop_offset += self.alph_size ** wl
		return words_idxs, words_valid


	def transform_all_subsequence(self, observations, time_stamps, windows):
		bop = np.zeros(self.bop_size * len(windows))
		pword = np.ones((len(self.features), len(self.word_length))) * -1
		offset = 0
		for window, window_step in zip(windows, windows_step):
			i = 0
			j = 1
			n = observations.size
			ini_obs = time_stamps[0]
			end_obs = time_stamps[n-1]
			break_points = None
			if self.global_break_points:
				break_points = self.get_break_points(observations)
			while i < n - self.tol and time_stamps[i] < end_obs:
				ini_time = time_stamps[i]
				end_time = ini_time + window
				while time_stamps[j] < end_time:
					if j == n-1:
						break
					j += 1
				if j - i > self.tol:
					if break_points is None:
						break_points = self.get_break_points(observations[i:j])
					word_idx, words_valid = self.sequence_to_word(observations, time_stamps, i, j, ini_time, end_time, break_points)
					for k in range(len(word_idx)):
						if words_valid[k]:
							if pword[k] != word_idx[k]:
								bop[word_idx[k] + offset] += 1
								pword[k] = word_idx[k]
				i += 1
			offset += self.bop_size
		return bop