import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from scipy import stats
from collections import defaultdict


class DTWBase(object):
	'''
	dtw algorithm basic implementation
	'''
	def __init__(self, arr1, arr2):
		self.arr1 = arr1
		self.arr2 = arr2
		self.n = len(arr1)
		self.m = len(arr2)
		self.warping_matrix = None
	def set_matrix(self):
		self.warping_matrix = np.zeros((self.n+1, self.m+1)) - 1
		self.warping_matrix[:,0] = np.inf
		self.warping_matrix[0,:] = np.inf
		self.warping_matrix[0,0] = 0
	def cost(self, i, j):
		return (self.arr1[i] - self.arr2[j]) ** 2
	def compute(self):
		for i in range(1,self.n+1):
			for j in range(1, self.m+1):
				d = self.cost(i-1, j-1)
				self.warping_matrix[i,j] = d + min(self.warping_matrix[i-1,j],
												   self.warping_matrix[i,j-1], 
												   self.warping_matrix[i-1,j-1])
	def find_path(self):
		n,m = self.warping_matrix.shape
		i = n-1
		j = m-1
		path = []
		while i != 0 and j != 0:
			path.append((i-1, j-1))
			vals = [self.warping_matrix[i-1,j], 
					self.warping_matrix[i,j-1], 
					self.warping_matrix[i-1,j-1]]
			idx = np.argmin(vals)
			if vals[0] == vals[1] and vals[0] == vals[2]:
				i -= 1
				j -= 1
			else:
				if idx == 0:
					i -= 1
				elif idx == 1:
					j -= 1
				elif idx == 2:
					i -= 1
					j -= 1
		return path[::-1]
	
	def distance(self):
		return self.warping_matrix[self.n,self.m]
	
	def paths(self):
		return self.warping_matrix



class IrregularDTW(DTWBase):
	'''
	Adapted DTW to handle empty segments represented by null values
	'''
	def __init__(self, arr1, arr2):
		super().__init__(arr1, arr2)
		self.valid_path_matrix = None
		
	def set_matrix(self):
		super(IrregularDTW, self).set_matrix()
		self.valid_path_matrix = np.ones(self.warping_matrix.shape)
		
	
	def cost(self, i, j):
		d = 0
		if self.arr1[i] is not None and self.arr2[j] is not None:
			for k in range(2):
				d += (self.arr1[i][k] - self.arr2[j][k]) ** 2
		return d
	
	def path_step_status(self, i, j):
		c1 = self.arr1[i] is None
		c2 = self.arr2[j] is None
#		 if (c1 and not c2) or (not c1 and c2):
		if c1 or c2:
			return 0
		else:
			return 1
		
	def get_prev_valid_i(self, i, j):
		prev_i = i - 1
		while not self.valid_path_matrix[prev_i, j]:
			prev_i -= 1
		return prev_i
	
	def get_prev_valid_j(self, i, j):
		prev_j = j - 1
		while not self.valid_path_matrix[i, prev_j]:
			prev_j -= 1
		return prev_j
	
	def get_prev_valid_diag(self, i_ini, i_end, j_ini, j_end):	
		i = i_end - 1 
		highest_pair = [0,0]
		while i >= i_ini:
			j = j_end - 1
			while j >= j_ini:
				if self.valid_path_matrix[i,j]:
					if i + j >= highest_pair[0] + highest_pair[1]:
						highest_pair = [i, j]
				j -= 1
			i -= 1
		return highest_pair
	
	def compute(self):
		for i in range(1, self.n+1):
			for j in range(1, self.m+1):
				self.valid_path_matrix[i,j] = self.path_step_status(i-1, j-1)			  
		
		for i in range(1,self.n+1):
			for j in range(1, self.m+1):
				if self.valid_path_matrix[i,j]:
					prev_i = self.get_prev_valid_i(i, j)
					prev_j = self.get_prev_valid_j(i, j)
#					 prev_i2, prev_j2 = self.get_prev_valid_diag(prev_i,i,prev_j,j)
					prev_i2, prev_j2 = prev_i, prev_j
#				 print("Step: (%d,%d), prev_i: %d, prev_j: %d" % (i, j, prev_i, prev_j))
					d = self.cost(i-1, j-1)
					self.warping_matrix[i,j] = d + min(self.warping_matrix[prev_i,j],
												   self.warping_matrix[i,prev_j], 
												   self.warping_matrix[prev_i2,prev_j2])
				
	def find_path(self):
		n,m = self.warping_matrix.shape
		i = n-1
		j = m-1
		while self.arr1[i-1] is None:
			i -= 1
		while self.arr2[j-1] is None:
			j -= 1
			
		
		path = []
#		 print("start:", i, j, "value:", self.warping_matrix[i,j])
		while i!=0 and j!=0:
			print(i, j, self.warping_matrix[i,j])
			if True:
				path.append((i-1, j-1))
				
			prev_i = self.get_prev_valid_i(i, j)
			prev_j = self.get_prev_valid_j(i, j)
#			 prev_i2, prev_j2 = self.get_prev_valid_diag(prev_i, i, prev_j, j)
			prev_i2, prev_j2 = prev_i, prev_j
				
#			 print("current: ", i, j)
#			 print("-->[i, prev_j]=[%d,%d], val: %s" % (i, prev_j, str(self.warping_matrix[i, prev_j])))
#			 print("-->[prev_i,j]=[%d,%d], val: %s" %(prev_i, j, str(self.warping_matrix[prev_i, j])))
#			 print("-->[prev_i,prev_j]=[%d,%d], val: %s" % (prev_i2, prev_j2, str(self.warping_matrix[prev_i2, prev_j2])))
			
			vals = [self.warping_matrix[prev_i,j], 
					self.warping_matrix[i,prev_j], 
					self.warping_matrix[prev_i2,prev_j2]]
			idx = np.argmin(vals)
			if vals[0] == vals[1] and vals[0] == vals[2]:
				i = prev_i2
				j = prev_j2
			else:
				if idx == 0:
					i = prev_i
				elif idx == 1:
					j = prev_j
				elif idx == 2:
					i = prev_i2
					j = prev_j2
		return path[::-1]
	
	def plot_path(self, window):
		fig = plt.figure(constrained_layout=False, figsize=(10, 6))
		gs = fig.add_gridspec(11, 1)
		ax1 = fig.add_subplot(gs[0:4, :])
		ax1.xaxis.tick_top()
		n = len(self.arr1)
		ticks = []
		tickslabels = []
		for i in range(n):
			ini_time = i*window
			end_time = (i+1)*window
			if self.arr1[i] is not None:
				s, v = self.arr1[i]
				x = np.linspace(ini_time, end_time)
				
				v_time = ini_time + (end_time - ini_time) / 2
				intercep = v - v_time*s
				y = x*s + intercep
				ax1.plot(x, y, '-b', linewidth=2)
				ax1.axvspan(ini_time, end_time, alpha=0.3, color='green')
				ticks.append(v_time)
				tickslabels.append(i)
			else:
				ax1.axvspan(ini_time, end_time, alpha=0.1, color='black')
		
		ax1.set_xticks(ticks)
		ax1.set_xticklabels(tickslabels, minor=False)
		
		
		ax3 = fig.add_subplot(gs[7:11, :])
		
		n = len(self.arr2)
		ticks = []
		tickslabels = []
		for i in range(n):
			ini_time = i*window
			end_time = (i+1)*window
			if self.arr2[i] is not None:
				s, v = self.arr2[i]
				x = np.linspace(ini_time, end_time)
				
				v_time = ini_time + (end_time - ini_time) / 2
				intercep = v - v_time*s
				y = x*s + intercep
				ax3.plot(x, y, '-b', linewidth=2)
				ax3.axvspan(ini_time, end_time, alpha=0.3, color='green')
				ticks.append(v_time)
				tickslabels.append(i)
			else:
				ax3.axvspan(ini_time, end_time, alpha=0.1, color='black')
		
		ax3.set_xticks(ticks)
		ax3.set_xticklabels(tickslabels, minor=False)
		
		
		n = max(len(self.arr1), len(self.arr2))
		
		ax1.set_xlim([0, n*window])
		ax3.set_xlim([0, n*window])
		
		ax2 = fig.add_subplot(gs[4:7, :])
		axR = ax2.twiny()
		ax2.set_xlim([0, n*window])
		axR.set_xlim([0, n*window])
		ax2.set_ylim([0, 1])
		ax2.axes.get_yaxis().set_visible(False)
		ax2.set_xticks(np.arange(n) * window  + window / 2)
		axR.set_xticks(np.arange(n) * window  + window / 2)
		ax2.set_xticklabels([])
		axR.set_xticklabels([])
		
		path = self.find_path()
		for k in range(len(path)):
			i,j = path[k]
			ax2.plot([j*window + window/2, i*window + window /2], [0, 1], color="orange")




class MultiBandDTW(IrregularDTW):
    def __init__(self, arr1, arr2, n_bands=6):
        super().__init__(arr1, arr2)
        self.n_bands = n_bands

    def path_step_status(self, i, j):
        for b in range(self.n_bands):
            pair1 = self.arr1[i][b]
            pair2 = self.arr2[i][b]
            if all(~np.isnan(pair1)) and all(~np.isnan(pair2)):
                return 1
        return 0

    def cost(self, i, j):
        d = 0
        c_bands = 0
        for b in range(self.n_bands):
            pair1 = self.arr1[i][b]
            pair2 = self.arr2[i][b]
            if all(~np.isnan(pair1)) and all(~np.isnan(pair2)):
                c_bands += 1
                for k in range(2):
                    d += (pair1[k] - pair2[k]) ** 2

        if c_bands == 0:
            raise ValueError("c_bands shouldnt be 0")

        return d / c_bands



        
        