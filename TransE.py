#Copyright (C) 2018  Seyed Mehran Kazemi, Licensed under the GPL V3; see: <https://www.gnu.org/licenses/gpl-3.0.en.html>
from tensor_factorizer import *
from reader import *

class TransE(TensorFactorizer):

	def __init__(self, params, dataset="wn18"):
		TensorFactorizer.__init__(self, model_name="TransE", loss_function="margin", params=params, dataset=dataset)

	def setup_weights(self):
		sqrt_size = 6.0 / math.sqrt(self.params.emb_size)
		self.rel_emb = tf.get_variable(name="rel_emb", initializer=tf.random_uniform(shape=[self.num_rel, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size))
		self.ent_emb = tf.get_variable(name="ent_emb", initializer=tf.random_uniform(shape=[self.num_ent, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size))
		self.var_list = [self.rel_emb, self.ent_emb]

	def define_regularization(self):
		self.regularizer = tf.reduce_sum(tf.nn.relu(tf.square(tf.norm(self.ent_emb, axis=1, ord=2)) - 1.0))

	def gather_train_embeddings(self):
		self.ph_emb = tf.gather(self.ent_emb, self.ph)
		self.pt_emb = tf.gather(self.ent_emb, self.pt)
		self.nh_emb = tf.gather(self.ent_emb, self.nh)
		self.nt_emb = tf.gather(self.ent_emb, self.nt)
		self.r_emb  = tf.gather(self.rel_emb, self.r)

	def gather_test_embeddings(self):
		self.h_emb = tf.gather(self.ent_emb, self.head) 
		self.r_emb = tf.gather(self.rel_emb, self.rel) 
		self.t_emb = tf.gather(self.ent_emb, self.tail) 

	def create_train_model(self):
		self.pos_dissims = tf.norm(self.ph_emb + self.r_emb - self.pt_emb, axis=1, ord=self.params.p_norm)
		self.neg_dissims = tf.norm(self.nh_emb + self.r_emb - self.nt_emb, axis=1, ord=self.params.p_norm)

	def create_test_model(self):
		self.dissims = tf.norm(self.h_emb + self.r_emb - self.t_emb, axis=1, ord=self.params.p_norm)
