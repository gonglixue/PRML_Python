import tensorflow as tf
import numpy as np

def cons(x):
    return tf.constant(x, dtype=tf.float32)

def compute_hessian(f, vars):
    mat = []
    for v1 in vars:
        temp = []
        for v2 in vars:
            # compute derivative twice, first w.r.t v2 and then w.r.t v1
            temp.append(tf.gradients(tf.gradients(f, v2)[0], v1)[0])

        temp = [cons[0] if t==None else t for t in temp]    # tf return None when there is no gradient
        temp = tf.stack(temp)
        mat.append(temp)

    mat = tf.stack(mat)
    return mat

x = tf.Variable(np.random.random_sample(), dtype=tf.float32)
y = tf.Variable(np.random.random_sample(), dtype=tf.float32)

func = tf.pow(x, cons(2)) + cons(2)*x*y + cons(3)*tf.pow(y,cons(2)) + cons(4)*x + cons(5)*y + cons(6)

hessian = compute_hessian(func, [x, y])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(hessian))
