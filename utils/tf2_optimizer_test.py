import tensorflow as tf 

op=tf.keras.optimizers.Adam(learning_rate=0.01,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-07,
                            amsgrad=False,
                            name='Adam')

x = tf.constant([2.0])
w = tf.Variable([7.0])
with tf.GradientTape(persistent=True) as cal_tape:
    # cal_tape.watch(w)
    y = x*w 
for _ in range(1000):
    gd =  cal_tape.gradient(y,[w])

    op.apply_gradients(zip(gd,[w]))
    print(w.numpy(),x.numpy())




