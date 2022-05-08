import os
import sys
import argparse
import subprocess

def base_call(method, use_sparse, use_metadata, metric, scale, tag, subset):
	s = ["python", "scripts/map.py", "plasticc_augment_v3", "plasticc_test", "--num_chunks", "25", 
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
	skip = [1, 2]
	c = 0
	for scale in [True, False]:
		for metric in ["cosine", "euclidean"]:
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

def main_umap():
	metric = "cosine"
	use_sparse = False
	method = "IBOPF"
	tag = "features_v3_UMAP_0.000000_hellinger_50"
	subset = False
	for scale in [True, False]:
		for metric in ["cosine", "euclidean"]:
			for use_metadata in [True, False]:
				s = base_call(method, use_sparse, use_metadata, metric, scale, tag, subset)
				ok = launch_subprocess([s])
				if not ok:
					return

def main_combined():
	metric = "cosine"
	use_sparse = False
	method = "IBOPF"
	tag 

if __name__ == '__main__':
	# main()
	main_umap()
