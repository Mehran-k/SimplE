#Copyright (C) 2018  Seyed Mehran Kazemi, Licensed under the GPL V3; see: <https://www.gnu.org/licenses/gpl-3.0.en.html>
from tensor_factorizer import *
from reader import *

class ComplEx(TensorFactorizer):

	def __init__(self, params, dataset="wn18"):
		TensorFactorizer.__init__(self, model_name="ComplEx", loss_function="likelihood", params=params, dataset=dataset)

	def setup_weights(self):
		sqrt_size = 6.0 / math.sqrt(self.params.emb_size)
		self.rel_emb_real = tf.get_variable(name="rel_emb_real", dtype=tf.float64, initializer=tf.random_uniform(shape=[self.num_rel, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.rel_emb_img  = tf.get_variable(name="rel_emb_img",  dtype=tf.float64, initializer=tf.random_uniform(shape=[self.num_rel, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.ent_emb_real = tf.get_variable(name="ent_emb_real", dtype=tf.float64, initializer=tf.random_uniform(shape=[self.num_ent, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.ent_emb_img  = tf.get_variable(name="ent_emb_img",  dtype=tf.float64, initializer=tf.random_uniform(shape=[self.num_ent, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.var_list = [self.rel_emb_real, self.rel_emb_img, self.ent_emb_real, self.ent_emb_img]

	def define_regularization(self):
		self.regularizer = (tf.nn.l2_loss(self.rel_emb_real) + tf.nn.l2_loss(self.rel_emb_img) + tf.nn.l2_loss(self.ent_emb_real) + tf.nn.l2_loss(self.ent_emb_img)) / self.num_batch

	def gather_train_embeddings(self):
		self.head_real = tf.gather(self.ent_emb_real, self.head)
		self.head_img  = tf.gather(self.ent_emb_img,  self.head)
		self.rel_real  = tf.gather(self.rel_emb_real, self.rel)
		self.rel_img   = tf.gather(self.rel_emb_img,  self.rel)
		self.tail_real = tf.gather(self.ent_emb_real, self.tail)
		self.tail_img  = tf.gather(self.ent_emb_img,  self.tail)

	def gather_test_embeddings(self):
		self.gather_train_embeddings()

	def create_train_model(self):
		self.dot1 = tf.reduce_sum(tf.multiply(self.rel_real, tf.multiply(self.head_real, self.tail_real)), 1)
		self.dot2 = tf.reduce_sum(tf.multiply(self.rel_real, tf.multiply(self.head_img, self.tail_img)), 1)
		self.dot3 = tf.reduce_sum(tf.multiply(self.rel_img, tf.multiply(self.head_real, self.tail_img)), 1)
		self.dot4 = tf.reduce_sum(tf.multiply(self.rel_img, tf.multiply(self.head_img, self.tail_real)), 1)
		self.init_scores = self.dot1 + self.dot2 + self.dot3 - self.dot4
		self.scores = tf.clip_by_value(self.init_scores, -20, 20)
		self.labels = self.y

	def create_test_model(self):
		self.dot1 = tf.reduce_sum(tf.multiply(self.rel_real, tf.multiply(self.head_real, self.tail_real)), 1)
		self.dot2 = tf.reduce_sum(tf.multiply(self.rel_real, tf.multiply(self.head_img, self.tail_img)), 1)
		self.dot3 = tf.reduce_sum(tf.multiply(self.rel_img, tf.multiply(self.head_real, self.tail_img)), 1)
		self.dot4 = tf.reduce_sum(tf.multiply(self.rel_img, tf.multiply(self.head_img, self.tail_real)), 1)
		self.init_scores = self.dot1 + self.dot2 + self.dot3 - self.dot4
		self.dissims = -tf.clip_by_value(self.init_scores, -20, 20)
