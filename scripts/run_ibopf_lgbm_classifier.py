import os
import sys
import argparse
import subprocess

base_train = ["python", "scripts/ibopf_lgbm_train_classifier.py", "plasticc_augment_v3", "--use_metadata"]
base_predict = ["python", "scripts/ibopf_lgbm_predict.py", "plasticc_test", "--num_chunks", "25"]


# def call_train(tag, args):
# 	s = ["python", "scripts/ibopf_lgbm_train_classifier.py", "plasticc_augment_v3", "--classifier", args.classifier, "--tag", tag]
# 	if args.use_metadata:
# 		s.append("--use_metadata")
# 	return s

# def call_predict(tag, args):
# 	s = ["python", "scripts/ibopf_lgbm_predict.py", "plasticc_test", "--classifier", args.classifier, "--tag", tag, "--num_chunks", str(args.num_chunks)]
# 	return s

def calls(classifier, tag):
	local_add = [
		"--classifier", classifier, 
		"--tag", tag]

	train_call = base_train.copy()
	train_call.extend(local_add)

	predict_call = base_predict.copy()
	predict_call.extend(local_add)

	return train_call, predict_call


def launch_subprocess(cc):
	for c in cc:
		print(" ".join(c))
		try:
			subprocess.check_call(c)
		except subprocess.CalledProcessError:
			print("################################################")
			print("error while runing process, ending")
			print("################################################")
			return False
	return True


def main_umap(args):
	# umap run grid-cases (densmap, supervised, n_neighbors, metric, min_dist, n_components)
	count = 0
	for densmap in [False]:
		for supervised in [False]:
			for n_neighbors in [100]:
				for metric in ["hellinger"]:
					for min_dist in [0.0]:
						for n_components in [5, 10, 20, 50]:
							name_f = "UMAP"
							if supervised:
								name_f += "_supervised"
							if min_dist is not None:
								name_f += "_%f" % min_dist
							if metric is not None:
								name_f += "_%s" % metric
							if densmap:
								name_f += "_densmap"
							tag = "features_v3_%s_%s" % (name_f, n_components)

							c1, c2 = calls(args.classifier, tag)
							if False:
								cc = [c2]
							else:
								cc = [c1, c2]
							ok = launch_subprocess(cc)
							count += 1
							if not ok:
								# we stop the experiment
								return

def main_combined(args):
	skip = [1, 2, 3, 4, 5, 6, 7, 9]
	c = 0
	for n_components in [2, 10, 20, 50, 100]:
		for method in ["UMAP", "LSA"]:
			c += 1
			if c in skip:
				continue
			if method == "UMAP":
				tag = "features_v3_UMAP_0.000000_cosinecombined_avocado_%d" % n_components
			elif method == "LSA":
				tag = "features_v3_LSAcombined_avocado_%d" % n_components
			else:
				tag = None
			c1, c2 = calls(args.classifier, tag)
			cc = [c1, c2]
			ok = launch_subprocess(cc)
			if not ok:
				# we stop the experiment
				return


def main_combined_post(args):
	for n_components in [2, 10, 30, 50, 100]:
		tag = "features_v3_LSA_%d" % n_components
		c1, c2 = calls(args.classifier, tag)
		c1.append("--combine_avocado")
		c2.append("--combine_avocado")
		cc = [c1, c2]
		ok = launch_subprocess(cc)
		if not ok:
			# we stop the experiment
			return


def main_manova(args):
	for n_components in [1, 10, 20, 30, 40, 50, 60]:
		tag = "features_v3_MANOVA_%d" % n_components
		c1, c2 = calls(args.classifier, tag)
		cc = [c1, c2]
		ok = launch_subprocess(cc)
		if not ok:
			# we stop the experiment
			return



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		'--classifier',
		help='Name of the classifier to produce.',
		default="lgbm"
	)
	args = parser.parse_args()

	# un-comment the ones you want you run
	# main_umap(args)
	# main_combined(args)
	# main_manova(args)
	main_combined_post(args)

