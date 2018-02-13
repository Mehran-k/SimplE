#Copyright (C) 2018  Seyed Mehran Kazemi, Licensed under the GPL V3; see: <https://www.gnu.org/licenses/gpl-3.0.en.html>
class Params:

	def __init__(self):
		pass

	def set_values(self, lr=-1, gamma=-1, bsize=-1, max_itr=-1, p_norm=-1, emb_size=-1, alpha=-1, neg_ratio=-1, save_after=-1, save_each=-1):
		self.learning_rate = lr
		self.gamma = gamma #the margin used in margin-based loss functions
		self.batch_size = bsize
		self.max_iterate = max_itr
		self.p_norm = p_norm #the p_norm (1 or 2) used in margin-based loss functions
		self.emb_size = emb_size
		self.alpha = alpha #the regularization parameter
		self.neg_ratio = neg_ratio
		self.save_each = save_each
		self.save_after = save_after

	def use_default(self, dataset, model):
		if dataset == "wn18" and model == "SimplE_ignr":
			self.set_values(lr=0.1, alpha=0.001, bsize=1415, max_itr=1000, emb_size=200, neg_ratio=1, save_after=50, save_each=50)
		elif dataset == "wn18" and model == "SimplE_avg":
			self.set_values(lr=0.1, alpha=0.03, bsize=1415, max_itr=1000, emb_size=200, neg_ratio=1, save_after=50, save_each=50)
		elif dataset == "wn18" and model == "ComplEx":
			self.set_values(lr=0.1, alpha=0.03, bsize=1415, max_itr=1000, emb_size=150, neg_ratio=1, save_after=50, save_each=50)
		elif dataset == "wn18" and model == "TransE":
			self.set_values(lr=0.001, alpha=0.25, gamma=2.0, p_norm=1, bsize=1415, max_itr=2000, emb_size=50, save_after=50, save_each=50)

		elif dataset == "fb15k" and model == "SimplE_ignr":
			self.set_values(lr=0.05, alpha=0.03, bsize=4832, max_itr=1000, emb_size=200, neg_ratio=10, save_after=100, save_each=100)
		elif dataset == "fb15k" and model == "SimplE_avg":
			self.set_values(lr=0.05, alpha=0.1, bsize=4832, max_itr=1000, emb_size=200, neg_ratio=10, save_after=100, save_each=100)
		elif dataset == "fb15k" and model == "ComplEx":
			self.set_values(lr=0.05, alpha=0.01, bsize=4832, max_itr=1000, emb_size=200, neg_ratio=10, save_after=100, save_each=100)			
		elif dataset == "fb15k" and model == "TransE":
			self.set_values(lr=0.001, alpha=0.25, gamma=1.0, p_norm=1, bsize=4832, max_itr=2000, emb_size=50, save_after=100, save_each=100)

		else:
			self.set_values(lr=0.1, alpha=0.001, gamma=2.0, p_norm=1, bsize=1415, max_itr=1000, emb_size=200, neg_ratio=1, save_after=50, save_each=50)

	def get_early_stopping_itrs(self):
		self.es_itrs = []
		i = self.save_after
		while i <= self.max_iterate:
			self.es_itrs.append(str(i))
			i += self.save_each
		return self.es_itrs
