import tensorflow as tf

from components.module import fill_in_missing


class ExecutorTest(tf.test.TestCase):
    def setUp(self):
        super(ExecutorTest, self).setUp()

    def test_fill_in_missing_dense_tensor(self):

        dense_tensor = tf.constant([[""], ["wow"], ["test"]])
        expected_tensor = tf.constant(["", "wow", "test"])
        rs = fill_in_missing(dense_tensor)
        comparison = tf.reduce_all(tf.equal(rs, expected_tensor))
        self.assertTrue(comparison)

    def test_fill_in_missing_sparse_tensor(self):

        # test_data = tf.constant(["", "test"])
        sparse_tensor = tf.SparseTensor(
            indices=[[1, 0], [2, 0]], values=["wow", "test"], dense_shape=[3, 1]
        )
        expected_tensor = tf.constant(["", "wow", "test"])
        rs = fill_in_missing(sparse_tensor)
        comparison = tf.reduce_all(tf.equal(rs, expected_tensor))
        self.assertTrue(comparison)
