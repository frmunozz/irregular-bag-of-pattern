import numpy as np
from scipy import stats


def get_slope(times, fluxes):
	slope, intercept, r_value, p_value, std_err = stats.linregress(times,fluxes)
	return slope


def get_value(fluxes):
	return np.mean(fluxes)


def mb_tva(times_mb, fluxes_mb, window, n_bands=6):
	i = 0
	j = 1
	mb_tva_vec = []
	mb_tva_arr = []
	ini_time = 0
	end_time = window
	widths = np.array([x[-1] - x[0] for x in times_mb])
	i_mb = np.zeros(n_bands)
	j_mb = np.zeros(n_bands)
	idx = np.argmax(widths)
	while ini_time <  times_mb[idxs][-1]:

		mb_vec = np.full((n_bands, 3), np.nan)

		atleast_1_valid = False
		for b in range(n_bands):
			i = i_mb[b]
			j = j_mb[b]
			while times[b][i] < ini_time:
				i += 1

			j = i
			while j < len(times[b]) and times[b][j] <= end_time:
				j += 1

			if j-i >= 1:
				atleast_1_valid = True
				time_seg = times[b][i:j]
				fluxes_seg = fluxes[b][i:j]
				if j-i > 1:
					slope = get_slope(time_seg, fluxes_seg)
				else:
					slope = 0
				val = get_value(fluxes_seg)
				mb_vec[b] = [slope, val, ini_time]
				# tva_vec.append((True,(slope, val, i, j, ini_time, end_time)))
			# else:
				# tva_vec.append((False, (ini_time, end_time)))
			i_mb[b] = j-1
			j_mb[b] = j

		if atleast_1_valid:
			mb_tva_vec.append((True, mb_vec))
			mb_tva_arr.append(mb_vec[:,:2])

		ini_time += window
		end_time += window
		i = j-1

	return tva_vec