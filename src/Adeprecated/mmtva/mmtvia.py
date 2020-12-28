import numpy as np
from scipy import stats


def get_slope(times, fluxes):
	slope, intercept, r_value, p_value, std_err = stats.linregress(times,fluxes)
	return slope


def get_value(fluxes):
	return np.mean(fluxes)


def mb_tvia(times, fluxes, window, threshold=2, n_bands=6):
	tva_vec = []
	ini_time = 0
	end_time = ini_time + window
	i_mb = np.zeros(n_bands)
	k_mb = np.zeros(n_bands)
	sizes = np.array([len(x) for x in fluxes])
	widths = np.array([x[-1] - x[0] for x in times])
	idx = np.argmax(widths)
	while k_mb + threshold <= len(fluxes):
		while times[k] <= end_time:
			if k == len(fluxes)-1:
				break
			k += 1
		if k - i >= threshold:
			time_seg = times[i:k]
			fluxes_seg = fluxes[i:k]
			if threshold > 1:
				slope = get_slope(time_seg, fluxes_seg)
			else:
				slope = 0
			val = get_value(fluxes_seg)
			tva_vec.append((True, (slope, val, i, k, ini_time, end_time)))
		ini_time = times[k]
		end_time = ini_time + window
		i = k
		k += 1

	return tva_vec

