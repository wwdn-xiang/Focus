import tensorflow as tf

so_file = './my_add.so'

class MyAddTest(tf.test.TestCase):
  def testMyAdd(self):
    my_add_module = tf.load_op_library(so_file)
    with self.test_session():
      result = my_add_module.my_add([5, 4, 3, 2, 1],[1, 2, 3, 4, 5])
      self.assertAllEqual(result.eval(), [0, 6, 6, 6, 6])

if __name__ == "__main__":
  #tf.test.main()
  my_add_module = tf.load_op_library(so_file)
  out = my_add_module.my_add([5, 4, 3, 2, 1],[1, 2, 3, 4, 5])
  sess = tf.Session()
  result = sess.run(out)
  print(result)
  #output [0, 6, 6, 6, 6]
