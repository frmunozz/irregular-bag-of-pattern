from pacmap import PaCMAP as old_pacmap

class PaCMAP(old_pacmap):

	def transform(self, x, basis=None, init=None, save_pairs=True):
		r = super(PaCMAP, self).transform(x, basis=basis, init=init, save_pairs=save_pairs)
		self.tree = None
		return r

	def fit_transform(self, x, init=None, save_pairs=None):
		r = super(PaCMAP, self).fit_transform(x, init=init, save_pairs=save_pairs)
		self.tree = None
		return r