import os
import sys
import argparse
import subprocess

base_fit = ["python", "scripts/ibopf_fit.py", "plasticc_augment_v3",
 			"--tag", "features_v3", "--load_sparse"]
base_transform = ["python", "scripts/ibopf_transform.py", "plasticc_test", 
				  "--tag", "features_v3", "--load_sparse"]

def call_fit_train(compact_method, n_components, n_neighbors, supervised):
	s = ["python", "scripts/ibopf_fit.py", "plasticc_augment_v3", 
		 "optimal_config_lsa.json", "--tag", "features_v3", 
		 "--compact_method", str(compact_method), 
		 "--n_components", str(n_components),
		 "--n_neighbors", str(n_neighbors),
		 "--load_sparse"]
	if supervised:
		s.append("--supervised")
	return s

def call_transform_test(compact_method, n_components, supervised):
	s = ["python", "scripts/ibopf_transform.py", "plasticc_test",
		 "optimal_config_lsa.json", "--num_chunks", "25", "--tag", "features_v3", 
		 "--compact_method", str(compact_method),
		 "--n_components", str(n_components),
		 "--load_sparse", "--subset"]
	if supervised:
		s.append("--supervised")
	return s

def gen_and_update_calls(local_add, args):
	fit_call = base_fit.copy()
	fit_call.extend(local_add)

	transform_call = base_transform.copy()
	transform_call.extend(local_add)
	if not args.fullset:
		transform_call.append("--subset")
	transform_call.extend(["--num_chunks", str(args.num_chunks)])

	return fit_call, transform_call


def calls_umap(densmap, supervised, n_neighbors, metric, min_dist, n_components, args):
	local_add = [
		"--compact_method", "UMAP", 
		"--n_components", str(n_components),
		"--n_neighbors", str(n_neighbors),
		"--metric", metric,
		"--min_dist", str(min_dist)]
	if densmap:
		local_add.append("--densmap")
	if supervised:
		local_add.append("--supervised")

	return gen_and_update_calls(local_add, args)

def calls_lsa(n_components, args):
	local_add = ["--compact_method", "LSA", "--n_components", str(n_components)]
	return gen_and_update_calls(local_add, args)

def calls_manova(n_components, args):
	local_add = ["--compact_method", "MANOVA", "--n_components", str(n_components)]
	return gen_and_update_calls(local_add, args)


def launch_subprocess(cc):
	for c in cc:
		print(":::::::::::::::::::::::::::::::::::::::::>")
		print(" ".join(c))
		try:
			subprocess.check_call(c)
		except subprocess.CalledProcessError:
			print("################################################")
			print("error while runing process, ending")
			print("################################################")
			return False
	return True


def main_umpa(args):
	# umap run grid-cases (densmap, supervised, n_neighbors, metric, min_dist, n_components)
	count = 0
	for densmap in [False]:
		for supervised in [False]:
			for n_neighbors in [100]:
				for metric in ["hellinger"]:
					for min_dist in [0.0]:
						for n_components in [5, 10, 20, 50]:
							c1, c2 = calls_umap(densmap, supervised, n_neighbors, metric, min_dist, n_components, args)
							if False:
								cc = [c2]
							else:
								cc = [c1, c2]
							ok = launch_subprocess(cc)
							count += 1
							if not ok:
								# we stop the experiment
								return
								
def main_lsa(args):
	skip = []
	for n_components in [2, 5, 10, 20, 50, 70, 100, 150, 200, 250, 300, 361]:
		c1, c2 = calls_lsa(n_components, args)
		cc = [c1, c2]
		if n_components in skip:
			continue
		ok = launch_subprocess(cc)
		if not ok:
			# we stop the experiment
			return

def main_manova(args):
	skip = []
	for n_components in [1, 5, 10, 20, 30, 40, 50, 60]:
		c1, c2 = calls_manova(n_components, args)
		cc = [c1, c2]
		if n_components in skip:
			continue
		ok = launch_subprocess(cc)
		if not ok:
			# we stop the experiment
			return


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--fullset', action="store_true")
	parser.add_argument("--num_chunks", default=25, type=int)
	args = parser.parse_args()
	# main_lsa(args)
	main_manova(args)
	# main_umap(args)