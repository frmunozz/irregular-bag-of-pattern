import os
import sys
import argparse
import subprocess

def base_call(method, use_sparse, use_metadata, metric, scale, tag, subset, num_chunks=25):
	s = ["python", "scripts/map.py", "plasticc_augment_v3", "plasticc_test", "--num_chunks", str(num_chunks),
		 "--method", method,
		 "--metric", metric,
		 # "--k", str(k),
		 "--tag", tag]
	if use_sparse:
		s.append("--use_sparse")
	if use_metadata:
		s.append("--use_metadata")
	if scale:
		s.append("--scale")
	if subset:
		s.append("--subset")

	return s

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

def main():
	subset = True
	skip = [1, 2, 3, 6]
	c = 0
	for scale in [True]:
		for metric in ["euclidean", "cosine"]:
			for use_metadata in [True, False]:
				for use_sparse, method, tag in zip([False, True, False], 
												   ["IBOPF", "IBOPF", "AVOCADO"],
												   ["features_v3_LSA", "features_v3_LSA", "features_v1"]):
					c += 1
					if c in skip:
						continue
					s = base_call(method, use_sparse, use_metadata, metric, scale, tag, subset)
					ok = launch_subprocess([s])
					if not ok:
						return


def main_full():
	subset = False
	scale = True
	metric = "euclidean"
	use_metadata = True
	use_sparse = False
	for method, tag in zip(["IBOPF", "AVOCADO"], ["features_v3_LSA", "features_v1"]):
		s = base_call(method, use_sparse, use_metadata, metric, scale, tag, subset, num_chunks=100)
		ok = launch_subprocess([s])
		if not ok:
			return

def main_umap():
	metric = "cosine"
	use_sparse = False
	method = "IBOPF"
	tag = "features_v3_UMAP_0.000000_hellinger_50"
	subset = False
	for scale in [True]:
		for metric in ["euclidean"]:
			for use_metadata in [True, False]:
				s = base_call(method, use_sparse, use_metadata, metric, scale, tag, subset)
				ok = launch_subprocess([s])
				if not ok:
					return

def main_supervised_umap():
	metric = "cosine"
	use_sparse = False
	method = "IBOPF"
	tag = "features_v3_UMAP_supervised_30"
	subset = False
	for scale in [True]:
		for metric in ["cosine", "euclidean"]:
			for use_metadata in [True]:
				s = base_call(method, use_sparse, use_metadata, metric, scale, tag, subset)
				ok = launch_subprocess([s])
				if not ok:
					return

def main_combined_lsa():
	use_sparse = False
	method = "IBOPF"
	tag_pre = "features_v3_LSAcombined_avocado_100"
	tag_post = "features_v3_LSA_361"
	subset = False
	for combine_late, tag in zip([False, True], [tag_pre, tag_post]):
		for scale in [True]:
			for metric in ["euclidean"]:
				for use_metadata in [True]:
					s = base_call(method, use_sparse, use_metadata, metric, scale, tag, subset)
					if combine_late:
						s.append("--combine_avocado")
					ok = launch_subprocess([s])
					if not ok:
						return


if __name__ == '__main__':
	main()
	main_full()
	# main_umap()
	# main_combined_lsa()
	# main_supervised_umap()
