#Copyright (C) 2018  Seyed Mehran Kazemi, Licensed under the GPL V3; see: <https://www.gnu.org/licenses/gpl-3.0.en.html>
from tensor_factorizer import *
from reader import *

class SimplE_avg(TensorFactorizer):

	def __init__(self, params, dataset="wn18"):
		TensorFactorizer.__init__(self, model_name="SimplE_avg", loss_function="likelihood", params=params, dataset=dataset)

	def setup_weights(self):
		sqrt_size = 6.0 / math.sqrt(self.params.emb_size)
		self.rel_emb      = tf.get_variable(name="rel_emb",      initializer=tf.random_uniform(shape=[self.num_rel, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.rel_inv_emb  = tf.get_variable(name="rel_inv_emb",  initializer=tf.random_uniform(shape=[self.num_rel, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.ent_head_emb = tf.get_variable(name="ent_head_emb", initializer=tf.random_uniform(shape=[self.num_ent, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.ent_tail_emb = tf.get_variable(name="ent_tail_emb", initializer=tf.random_uniform(shape=[self.num_ent, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.var_list = [self.rel_emb, self.rel_inv_emb, self.ent_head_emb, self.ent_tail_emb]

	def define_regularization(self):
		self.regularizer = (tf.nn.l2_loss(self.ent_head_emb) + tf.nn.l2_loss(self.ent_tail_emb) + tf.nn.l2_loss(self.rel_emb) + tf.nn.l2_loss(self.rel_inv_emb)) / self.num_batch

	def gather_train_embeddings(self):
		self.h1_emb = tf.gather(self.ent_head_emb, self.head)
		self.h2_emb = tf.gather(self.ent_head_emb, self.tail)
		self.t1_emb = tf.gather(self.ent_tail_emb, self.tail)
		self.t2_emb = tf.gather(self.ent_tail_emb, self.head)
		self.r1_emb = tf.gather(self.rel_emb, self.rel)
		self.r2_emb = tf.gather(self.rel_inv_emb, self.rel)

	def gather_test_embeddings(self):
		self.gather_train_embeddings()

	def create_train_model(self):
		self.init_scores = (tf.reduce_sum(tf.multiply(tf.multiply(self.h1_emb, self.r1_emb), self.t1_emb), 1) + tf.reduce_sum(tf.multiply(tf.multiply(self.h2_emb, self.r2_emb), self.t2_emb), 1)) / 2.0
		self.scores = tf.clip_by_value(self.init_scores, -20, 20) #Without clipping, we run into NaN problems.
		self.labels = self.y

	def create_test_model(self):
		self.init_scores = (tf.reduce_sum(tf.multiply(tf.multiply(self.h1_emb, self.r1_emb), self.t1_emb), 1) + tf.reduce_sum(tf.multiply(tf.multiply(self.h2_emb, self.r2_emb), self.t2_emb), 1)) / 2.0
		self.dissims = -tf.clip_by_value(self.init_scores, -20, 20) #Without clipping, we run into NaN problems.
