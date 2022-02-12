from __future__ import division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import seaborn as sns


## Cargar las señales de la base de datos
labelsS1 = np.genfromtxt('labelsS1.csv', delimiter=',')
labelsS2 = np.genfromtxt('labelsS2.csv', delimiter=',')
labelsS3 = np.genfromtxt('labelsS3.csv', delimiter=',')

signalsS1 = np.genfromtxt('signalsS1.csv', delimiter=',')
signalsS2 = np.genfromtxt('signalsS2.csv', delimiter=',')
signalsS3 = np.genfromtxt('signalsS3.csv', delimiter=',')

## Cada set tiene diferentes tamaños
signalsS1 = np.reshape(signalsS1,[294,4,128])
signalsS2 = np.reshape(signalsS2,[300,4,128])
signalsS3 = np.reshape(signalsS3,[299,4,128])


#Aumentar renglones para la resolución
factor = 8
canales = 4 * factor
conjuntoS1 = np.array(np.zeros((signalsS1.shape[0],canales, 128),dtype=np.float))
conjuntoS2 = np.array(np.zeros((signalsS2.shape[0],canales, 128),dtype=np.float))
conjuntoS3 = np.array(np.zeros((signalsS3.shape[0],canales, 128),dtype=np.float))


def expand_datos(signals):
    conjunto = np.array(np.zeros((signals.shape[0],canales, 128),dtype=np.float))
    for i in range(0, signals.shape[0]):
        conjunto[i][0:8][:] = signals[i][0][:]
        conjunto[i][8:17][:] = signals[i][1][:]
        conjunto[i][17:25][:] = signals[i][2][:]
        conjunto[i][25:32][:] = signals[i][3][:]
    return conjunto
conjuntoS1 = expand_datos(signalsS1)
conjuntoS2 = expand_datos(signalsS2)
conjuntoS3 = expand_datos(signalsS3)

#SUJETO 1, AvsRE, de 0-96, AUvsUI 96-196, UIvsRE 198-296
# SUJETO 2 todas las señales son de 100 en 100
# SUJETO 3, es 0-99 y de ahi de 100 en 100

lim_inf = 0
lim_sup = 96
conjunto_entrenar = conjuntoS1[lim_inf:lim_sup][:][:]
etiquetas = labelsS1[lim_inf:lim_sup][:][:]

##### Quitar 3 comentarios siguientes para cuando se entrene en la clase 3: /a/:/u/. Para la segunda clase basta
##### con restarle 1 al renglón donde se define a etiquetas


# for l in range(etiquetas.shape[0]):
#     if etiquetas[l]==2:
#         etiquetas[l]=1


dimension_tensor=128
##### Inicio de implementacion de la entrada, primero la capa de entrada
X = tf.placeholder(shape=[None, canales, dimension_tensor, 1], dtype=tf.float32, name="X")

#### Primary Caps
caps1_n_maps = 8
caps1_n_caps = caps1_n_maps *8 * 56  # 1320 primary capsules
caps1_n_dims = 4
conv1_params = {
    "filters": 40,
    "kernel_size": 10,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu
}

conv2_params = {
    "filters": caps1_n_maps * caps1_n_dims, # convolutional filters
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}

   

conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
# conv3 = tf.layers.conv2d(conv2, name="conv3", **conv3_params)
caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],
                       name="caps1_raw")

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector

caps1_output = squash(caps1_raw, name="caps1_output")
# caps1_output = caps1_raw      

##########################################################
################################## Digit Caps
caps2_n_caps = 2
clases = caps2_n_caps
caps2_n_dims = 20 

init_sigma = 0.1
    
W_init = tf.random_normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")

batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")
caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                             name="caps1_output_tiled")

caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")

##### Routing by agreement
#####Ronda 1
raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")

routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                   name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                             name="weighted_sum")

caps2_output_round_1 = squash(weighted_sum, axis=-2,
                              name="caps2_output_round_1")

##### Ronda 2
caps2_output_round_1_tiled = tf.tile(
    caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
    name="caps2_output_round_1_tiled")
agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")

raw_weights_round_2 = tf.add(raw_weights, agreement,
                             name="raw_weights_round_2")

routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                        dim=2,
                                        name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                           caps2_predicted,
                                           name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                     axis=1, keep_dims=True,
                                     name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2,
                              axis=-2,
                              name="caps2_output_round_2")

caps2_output = caps2_output_round_2

##Probabilidad de cada clase
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)
y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")
y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")

y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

## MARGIN LOSS
m_plus = 0.7
m_minus = 0.1
lambda_ = 0.5
T = tf.one_hot(y, depth=caps2_n_caps, name="T")

caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_output_norm")
caps2_output_norm1 = tf.squeeze(caps2_output_norm)
present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, clases),
                           name="present_error")
absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, clases),
                          name="absent_error")
L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")
margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

################RECONSTRUCTION
mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")
reconstruction_targets = tf.cond(mask_with_labels, # condition
                                 lambda: y,        # if True
                                 lambda: y_pred,   # if False
                                 name="reconstruction_targets")
reconstruction_mask = tf.one_hot(reconstruction_targets,
                                 depth=caps2_n_caps,
                                 name="reconstruction_mask")

reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],
    name="reconstruction_mask_reshaped")

caps2_output_masked = tf.multiply(
    caps2_output, reconstruction_mask_reshaped,
    name="caps2_output_masked")

decoder_input = tf.reshape(caps2_output_masked,
                           [-1, caps2_n_caps * caps2_n_dims],
                           name="decoder_input")

##############DECODER
n_hidden1 = 1024
n_hidden2 = 2048
n_hidden3 = 4096
n_hidden4 = 8192
n_output = canales*dimension_tensor
with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
    # hidden3 = tf.layers.dense(hidden2, n_hidden3,
    #                           activation=tf.nn.relu,
    #                           name="hidden3")
    # hidden4 = tf.layers.dense(hidden3, n_hidden4,
    #                           activation=tf.nn.selu,
    #                           name="hidden4")
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")
    # decoder_output = tf.nn.dropout(decoder_output, rate=0.1, seed=1)

##################Reconstruction loss
X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(decoder_output - X_flat,
                               name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
                                     name="reconstruction_loss")

alpha = 0.0005

loss = tf.add(1*margin_loss, alpha * reconstruction_loss, name="loss")
correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
optimizer = tf.train.AdamOptimizer()
# optimizer = tf.train.ProximalAdagradOptimizer(
#     learning_rate=0.01, initial_accumulator_value=0.1, l1_regularization_strength=0.0,
#     l2_regularization_strength=0.0, use_locking=False,
#     name='ProximalAdagrad')
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()

confusion = np.array(np.zeros((2,2),dtype=np.float))
confusion1 = np.array(np.zeros((2,2),dtype=np.float))
kfold = KFold(n_splits=5, shuffle=True)
# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(conjunto_entrenar, etiquetas):
    X_train = conjunto_entrenar[train]
    X_validate = conjunto_entrenar[test]
    
    y_train = etiquetas[train]
    y_validate = etiquetas[test]
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)
    
    

    ##########training
    n_epochs = 10
    batch_size = 1
    restore_checkpoint = True
    
    n_iterations_per_epoch = int(X_train.shape[0]/batch_size)
    n_iterations_validation = int(X_validate.shape[0]/batch_size)
    n_iterations_test = int(X_test.shape[0]/batch_size)
    best_loss_val = np.infty
    checkpoint_path = "./my_capsule_network"

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(1, n_iterations_per_epoch + 1):
                limite_low = batch_size*(iteration-1)
                X_batch = X_train[limite_low:(batch_size*iteration)][:][:]
                X_batch_train = np.expand_dims(X_batch, axis=3)
                y_batch = np.reshape(y_train, y_train.shape[0])
                y_batch_train = y_batch[limite_low:(batch_size*iteration)]
                # Run the training operation and measure the loss:
                _, loss_train = sess.run(
                    [training_op, loss],
                    feed_dict={X: X_batch_train,
                                y: y_batch_train,
                                mask_with_labels: True})
                # print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                #           iteration, n_iterations_per_epoch,
                #           iteration * 100 / n_iterations_per_epoch,
                #           loss_train),
                #     end="")
        
            # At the end of each epoch,
            # measure the validation loss and accuracy:
            loss_vals = []
            acc_vals = []
            caps1_raw_vals = []
            for iteration in range(1, n_iterations_validation + 1):
                limite_low = batch_size*(iteration-1)
                X_batch_validate = X_validate[limite_low:(batch_size*iteration)][:][:]
                X_batch_validate = np.expand_dims(X_batch_validate, axis=3)
                y_batch = np.reshape(y_validate, y_validate.shape[0])
                y_batch_validate = y_batch[limite_low:(batch_size*iteration)]
                loss_val, acc_val, caps1_raw_val = sess.run(
                        [loss, accuracy, caps1_raw],
                        feed_dict={X:X_batch_validate,
                                    y: y_batch_validate})
                loss_vals.append(loss_val)
                acc_vals.append(acc_val)
                caps1_raw_vals.append(caps1_raw_val)
                # print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                #           iteration, n_iterations_validation,
                #           iteration * 100 / n_iterations_validation),
                #     end=" " * 10)
            loss_val = np.mean(loss_vals)
            acc_val = np.mean(acc_vals)
            # print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            #     epoch + 1, acc_val * 100, loss_val,
            #     " (improved)" if loss_val < best_loss_val else ""))
                
            # And save the model if it improved:
            if loss_val < best_loss_val:
                save_path = saver.save(sess, checkpoint_path)
                best_loss_val = loss_val
            
    ###############EVALUATION
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        loss_tests = []
        acc_tests = []
        margin_loss_train_values = []
        caps2_norm=[]
        resultado = []
        for iteration in range(1, n_iterations_test + 1):
            limite_low = batch_size*(iteration-1)
            X_batch_test = X_test[limite_low:(batch_size*iteration)][:][:]
            X_batch_test = np.expand_dims(X_batch_test, axis=3)
            y_batch = np.reshape(y_test, y_test.shape[0])
            y_batch_test = y_batch[limite_low:(batch_size*iteration)]
            loss_test, acc_test, margin_loss_train, caps2_output_norm_eval, y_pred_test = sess.run(
                    [loss, accuracy, margin_loss,caps2_output_norm1, y_pred],
                    feed_dict={X: X_batch_test,
                               y: y_batch_test})
            caps2_norm.append(caps2_output_norm_eval)
            margin_loss_train_values.append(margin_loss_train)
            loss_tests.append(loss_test)
            acc_tests.append(acc_test)
            
            resultado.append(y_pred_test)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_test,
                      iteration * 100 / n_iterations_test),
                end=" " * 10)
            loss_test = np.mean(loss_tests)
            acc_test = np.mean(acc_tests)
        print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
            acc_test * 100, loss_test))
        resultado1 = np.array(resultado)
        resultado1 = np.squeeze(resultado1)
        confusion = confusion_matrix(y_test, resultado1)
        confusion1 = confusion1 + confusion
        f1, ax1 = plt.subplots(figsize=(9,6))
        sns.heatmap(confusion, annot=True, linewidth=0.5, ax=ax1, cmap="Blues")
        ax1.set_title("Confusion Matrix for subject 1 -- /a/:control")
        plt.ylabel("Labels")
        plt.xlabel("Prediction")
        plt.show()
        
        print(y_test)
        print(resultado1)
        
        
f2, ax2 = plt.subplots(figsize=(9,6))
sns.heatmap(confusion1, annot=True, linewidth=0.5, ax=ax2, cmap="Greens")
ax2.set_title("Confusion Matrix for subject 1 -- /a/:control")
plt.ylabel("Prediction of CapsVI")
plt.xlabel("Labels")
        
plt.show()
        
        

    
    # ####################PREDICTIONS
    # n_samples = 5
    
    # sample_images = X_test[:][:][:]
    # sample_images = np.expand_dims(sample_images, axis=3)
    # with tf.Session() as sess:
    #     saver.restore(sess, checkpoint_path)
    #     caps1_output_value, conv1_value, caps2_output_value, decoder_output_value, y_pred_value, caps2_output_norm_value = sess.run(
    #         [caps1_output, conv1, caps2_output, decoder_output, y_pred, caps2_output_norm1],
    #             feed_dict={X: sample_images,
    #                         y: np.array([], dtype=np.int64)})
    #     gr = tf.get_default_graph()
    #     conv1_kernel=gr.get_tensor_by_name("conv1/kernel:0").eval()

    # reconstructions = decoder_output_value.reshape([-1, canales, dimension_tensor])

    # # for i in range(0,40):
        

    # plt.figure(figsize=(n_samples * 2, 3))
    # for index in range(n_samples):
    #     plt.subplot(3, n_samples //2, index + 1)
    #     imagen = sample_images[index][:][:]
    #     plt.imshow(imagen.reshape([canales,-1]))
    #     plt.title("Label:" + str(y_test[index]))
    #     plt.axis("off")

    # plt.show()

    # plt.figure(figsize=(n_samples * 2, 3))
    # for index in range(n_samples):
    #     plt.subplot(3, n_samples//2, index + 1)
    #     plt.title("Predicted:" + str(y_pred_value[index]))
    #     plt.imshow(reconstructions[index])
    #     plt.axis("off")
    
    # plt.show()

    # ########################INTERPRETING THE OUTPUT VECTORS
    # caps2_output_value.shape
    # def tweak_pose_parameters(output_vectors, min=-0.5, max=0.5, n_steps=11):
    #     steps = np.linspace(min, max, n_steps) # -0.25, -0.15, ..., +0.25
    #     pose_parameters = np.arange(caps2_n_dims) # 0, 1, ..., 15
    #     tweaks = np.zeros([caps2_n_dims, n_steps, 1, 1, 1, caps2_n_dims, 1])
    #     tweaks[pose_parameters, :, 0, 0, 0, pose_parameters, 0] = steps
    #     output_vectors_expanded = output_vectors[np.newaxis, np.newaxis]
    #     return tweaks + output_vectors_expanded

    # n_steps = 6

    # tweaked_vectors = tweak_pose_parameters(caps2_output_value, n_steps=n_steps)
    # tweaked_vectors_reshaped = tweaked_vectors.reshape(
    #     [-1, 1, caps2_n_caps, caps2_n_dims, 1])
    
    # # tweak_labels = y_test
    # # tweak_labels = np.reshape(tweak_labels, tweak_labels.shape[0])
    
    # tweak_labels = np.tile(y_test[:n_samples], caps2_n_dims * n_steps)
    
    # with tf.Session() as sess:
    #     saver.restore(sess, checkpoint_path)
    #     decoder_output_value = sess.run(
    #             decoder_output,
    #             feed_dict={caps2_output: tweaked_vectors_reshaped,
    #                     mask_with_labels: False,
    #                         y: tweak_labels})
    #     tweak_reconstructions = decoder_output_value.reshape(
    #         [caps2_n_dims, n_steps, n_samples, canales, dimension_tensor])

    # for dim in range(3):
    #     print("Tweaking output dimension #{}".format(dim))
    #     plt.figure(figsize=(n_steps / 1.2, n_samples / 1.5))
    #     for row in range(n_samples):
    #         for col in range(n_steps):
    #             plt.subplot(n_samples, n_steps, row * n_steps + col + 1)
    #             plt.imshow(tweak_reconstructions[dim, col, row])
    #             plt.axis("off")
    #     plt.show()
    # print("Y TEST")
    # print(y_test)
    # print("PREDICCION")
    # print(y_pred_value)
    
