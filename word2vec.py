"""Classes to load train word vectors on text data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import math

# Dependency imports
import numpy as np
import sonnet as snt
import tensorflow as tf
from tensorflow.python.platform import gfile

FLAGS = tf.flags.FLAGS

class TokenDataSource(object):
  """Encapsulates loading/tokenization logic for disk-based data."""

  UNK = "_unk_"
  DEFAULT_START_TOKENS = [UNK]

  def __init__(self, data_file, vocab_data_file, vocabulary_size=50000):
    """Creates a TokenDataSource instance.

    Args:
      data_file: file object containing text data to be tokenized.
      vocab_data_file: secondary file object containing text data used to initialize
        the vocabulary.
    """
    def reading_function(f):
      return list(f.read().split())

    self._vocab_dict = {}
    self._inv_vocab_dict = {}
    self.count = {}

    token_list = reading_function(vocab_data_file)

    token_list = [tok for (tok, _) in collections.Counter(token_list).most_common(vocabulary_size - 1)]
    self.vocab_size = 0

    for token in self.DEFAULT_START_TOKENS + token_list:
      if token not in self._vocab_dict:
        self._vocab_dict[token] = self.vocab_size
        self._inv_vocab_dict[self.vocab_size] = token
        self.vocab_size += 1

    raw_data = reading_function(data_file)
    self.flat_data = np.array(self.tokenize(raw_data), dtype=np.int32)
    self.num_tokens = self.flat_data.shape[0]

  def tokenize(self, token_list):
    """Produces the list of integer indices corresponding to a token list."""
    def token_and_count(token):
      encoded = self._vocab_dict.get(token, self._vocab_dict[self.UNK])
      count = self.count.get(encoded, 0)
      self.count[encoded] = count+1
      return encoded

    return [token_and_count(token) for token in token_list]

  def decode(self, token_list):
    """Produces a human-readable representation of the token list."""
    return " ".join([self._inv_vocab_dict[token] for token in token_list])


SequenceDataOpsNoMask = collections.namedtuple("SequenceDataOpsNoMask",
                                               ("obs", "target"))

class SkipGramDataset(snt.AbstractModule):
  """Skip gram text sequence data."""

  def __init__(self, data_file, vocab_data_file=None, skip_window=1, batch_size=1,
               name="skip_gram_text_dataset"):
    """Initializes a SkipGramDataset sequence data object.

    Args:
      data_file: path to file containing text data.
      skip_window: the size of the window to sample.
      batch_size: batch size.
      name: object name.
    """
    super(SkipGramDataset, self).__init__(name=name)

    # Generate vocab from train set.

    self._data_file = gfile.Open(data_file)
    self._vocab_data_file = gfile.Open(vocab_data_file or data_file)
    self._skip_window = skip_window
    self._batch_size = batch_size

    self._data_source = TokenDataSource(data_file=self._data_file, vocab_data_file=self._vocab_data_file)

    self._vocab_size = self._data_source.vocab_size
    self._flat_data = self._data_source.flat_data
    self._n_flat_elements = self._data_source.num_tokens
    self._count = self._data_source.count

    self._num_batches = self._n_flat_elements // (self._skip_window * batch_size)
    self._reset_head_indices()

    self._queue_capacity = 10

  @property
  def vocab_size(self):
    return self._vocab_size

  def _reset_head_indices(self):
    self._head_indices = np.arange(self._batch_size, dtype=np.int32)

  def _get_batch(self):
    """Returns a batch of skip grams.

    Returns:
      obs: np.int32 array of size [Time, Batch]
      target: np.int32 array of size [Time, Batch]
    """
    batch_indices = np.mod(
        np.array([
                   np.arange(head_index - self._skip_window, head_index + self._skip_window + 1)
                   for head_index in self._head_indices
                 ]),
        self._n_flat_elements)


    obs = np.array([
      self._flat_data[indices[i]]
      for i in range(0, self._skip_window) + range(self._skip_window+1, 1+self._skip_window*2)
      for indices in batch_indices
    ])

    target = np.array([
      self._flat_data[indices[i]]
      for i in range(self._skip_window * 2)
      for indices in batch_indices
    ])

    self._head_indices = np.mod(
        self._head_indices + 1, self._n_flat_elements)

    return obs, target

  def _build(self):
    """Returns a tuple containing observation and target tensors."""
    q = tf.FIFOQueue(
        self._queue_capacity, [tf.int32, tf.int32],
        shapes=[[self._skip_window*2*self._batch_size]]*2)
    obs, target = tf.py_func(self._get_batch, [], [tf.int32, tf.int32])
    enqueue_op = q.enqueue([obs, target])
    obs, target = q.dequeue()
    # needed for nce loss - expects 2d array, TODO move somewhere closer
    target = tf.reshape(target, (self._skip_window*2*self._batch_size, 1))
    tf.train.add_queue_runner(tf.train.QueueRunner(q, [enqueue_op]))
    return SequenceDataOpsNoMask(obs, target)

  def sample(self,sample_size, sample_window):
    """
    Args:
      sample_size: Random set of words to evaluate similarity on.
      sample_window: Only pick dev samples in the head of the distribution.
    """
    sample_values = np.random.choice(sample_window, sample_size, replace=False)
    return tf.constant(sample_values, dtype=tf.int32)


  def decode(self, token_list):
      return self._data_source.decode(token_list)

  def to_human_readable(self,
                        data,
                        label_batch_entries=True,
                        indices=None,
                        sep="\n"):
    """Returns a human-readable version of encoding of words.

    Args:
      data: A tuple with (obs, target). `obs` is a numpy array with encoding of words.
      label_batch_entries: bool. Whether to add numerical label before each
          batch element in the output string.
      indices: List of int or None. Used to select a subset of minibatch indices
          to print. None will print the whole minibatch.
      sep: A char separator which separates the output for each batch. Defaults
          to the newline character.

    Returns:
      String with the words from `data[0]`.
    """
    obs = data[0]
    batch_size = obs.shape[0]
    result = []
    indices = xrange(batch_size) if not indices else indices
    for b in indices:
      index_seq = [obs[b]]
      prefix = "b_{}: ".format(b) if label_batch_entries else ""
      result.append(prefix + self._data_source.decode(index_seq))
    return sep.join(result)

class Word2VecModel(snt.AbstractModule):
  """A Word2Vec model for use on SkipGramDataset."""

  def __init__(self, vocabulary_size, embedding_size, num_sampled = 64,
               name="word2vec_model"):
    """Constructs a `Word2VecModel`.

    Args:
      vocabulary_size: .
      embedding_size: .
      num_sampled: Number of negative examples to sample.
      name: Name of the module.
    """

    super(Word2VecModel, self).__init__(name=name)

    self._vocabulary_size = vocabulary_size
    self._embedding_size = embedding_size
    self._num_sampled = num_sampled
    self._batch_size = batch_size

  def _build(self, inputs):
    """Builds the word2vec sub graph.

    Args:
      inputs: .

    Returns: .
    """
    self._embeddings = tf.get_variable(
          "embeddings",
          dtype=tf.float32,
          shape=[self._vocabulary_size, self._embedding_size],
          initializer=tf.random_uniform_initializer(-1.0, 1.0))

    self._nce_weights = tf.get_variable(
          "nce_weights",
          dtype=tf.float32,
          shape=[self._vocabulary_size, self._embedding_size],
          initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(self._embedding_size)))

    self._nce_biases = tf.get_variable(
          "nce_biases",
          dtype=tf.float32,
          shape=[self._vocabulary_size],
          initializer=tf.zeros_initializer())

    self._embed = tf.nn.embedding_lookup(self._embeddings, inputs)

    norm = tf.sqrt(tf.reduce_sum(tf.square(self._embeddings), 1, keep_dims=True))

    self._normalized_embeddings = self._embeddings / norm

    return self._embed

  def cost(self, logits, target):
      return tf.reduce_mean(
          tf.nn.nce_loss(weights=self._nce_weights,
                         biases=self._nce_biases,
                         labels=target,
                         inputs=logits,
                         num_true=1,
                         num_sampled=self._num_sampled,
                         num_classes=self._vocabulary_size))

  def similarity(self, ids):
    embedded = tf.nn.embedding_lookup(self._normalized_embeddings, ids)
    return self._cosine_similarity(embedded)

  def _cosine_similarity(self, embeddings):
    return tf.matmul(embeddings, self._normalized_embeddings, transpose_b=True, name="cosine_similarity")

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
sample_size=16
sample_window=100
top_k = 8

#dataset = SkipGramDataset("/data/tiny-shakespeare.txt", batch_size=batch_size)
dataset = SkipGramDataset("/data/text8", batch_size=batch_size)
word2vec = Word2VecModel(dataset.vocab_size, embedding_size)

input_sequence, target_sequence = dataset()
sample_dataset = dataset.sample(sample_size, sample_window)

output_sequence_logits = word2vec(input_sequence)
loss = word2vec.cost(output_sequence_logits, target_sequence)
similarity = word2vec.similarity(sample_dataset)

# Construct the SGD optimizer using a learning rate of 1.0.
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

def learn():
  with tf.Session() as sess:
    # Create the op for initializing variables.
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    with sess.as_default():
      try:
        num_steps = 100001
        average_loss = 0

        for step in xrange(num_steps):
          # We perform one update step by evaluating the optimizer op (including it
          # in the list of returned values for session.run()
          _, loss_val = sess.run([optimizer, loss])
          average_loss += loss_val

          if step % 2000 == 0:
            if step > 0:
              average_loss /= 2000
              # The average loss is an estimate of the loss over the last 2000 batches.
              print('Average loss at step ', step, ': ', average_loss)
              average_loss = 0

          # Note that this is expensive (~20% slowdown if computed every 500 steps)
          if step % 10000 == 0:
            if step > 0:
              sim = sess.run(similarity)

              for i in xrange(sample_size):
                sample = dataset.decode([sample_dataset.eval()[i]])
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                similar = dataset.decode(nearest)
                print("%s -> %s" % (sample, similar))

      except tf.errors.OutOfRangeError:
        print('Done')
      finally:
        coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

learn()
