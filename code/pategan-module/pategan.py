import tensorflow as tf
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from keras import layers, Model

warnings.filterwarnings("ignore")

def pate_lambda(x, teacher_models, lamda):
    """Returns PATE_lambda(x).

    Args:
        - x: feature vector
        - teacher_models: a list of teacher models
        - lamda: parameter

    Returns:
        - n0, n1: the number of label 0 and 1, respectively
        - out: label after adding Laplace noise.
    """
    y_hat = [teacher.predict(np.reshape(x, [1, -1])) for teacher in teacher_models]
    y_hat = np.asarray(y_hat)
    n0 = np.sum(y_hat == 0)
    n1 = np.sum(y_hat == 1)

    lap_noise = np.random.laplace(loc=0.0, scale=lamda)
    out = (n1 + lap_noise) / float(n0 + n1)
    out = int(out > 0.5)

    return n0, n1, out

def build_generator(z_dim, generator_h_dim, output_dim):
    """Builds the generator model."""
    z_input = layers.Input(shape=(z_dim,))
    x = layers.Dense(generator_h_dim, activation='tanh', kernel_initializer='glorot_normal')(z_input)
    x = layers.Dense(generator_h_dim, activation='tanh', kernel_initializer='glorot_normal')(x)
    x_out = layers.Dense(output_dim, activation='sigmoid', kernel_initializer='glorot_normal')(x)
    return Model(inputs=z_input, outputs=x_out)

def build_student(input_dim, student_h_dim):
    """Builds the student model."""
    x_input = layers.Input(shape=(input_dim,))
    x = layers.Dense(student_h_dim, activation='relu', kernel_initializer='glorot_normal')(x_input)
    x_out = layers.Dense(1, activation=None, kernel_initializer='glorot_normal')(x)
    return Model(inputs=x_input, outputs=x_out)

def pategan(x_train, parameters):
    """Basic PATE-GAN framework.

    Args:
        - x_train: training data
        - parameters: PATE-GAN parameters

    Returns:
        - x_train_hat: generated training data by differentially private generator
    """
    # Extract PATE-GAN parameters
    n_s = parameters['n_s']
    batch_size = parameters['batch_size']
    k = parameters['k']
    epsilon = parameters['epsilon']
    delta = parameters['delta']
    lamda = parameters['lamda']

    # Network parameters
    no, dim = x_train.shape
    z_dim = dim
    student_h_dim = dim
    generator_h_dim = 4 * dim

    # Partition data into `k` subsets
    partition_data_no = no // k
    idx = np.random.permutation(no)
    x_partition = [x_train[idx[i * partition_data_no:(i + 1) * partition_data_no]] for i in range(k)]

    # Build generator and student models
    generator = build_generator(z_dim, generator_h_dim, dim)
    student = build_student(dim, student_h_dim)

    # Optimizers
    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
    student_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)

    # Training loop variables
    epsilon_hat = 0
    alpha = np.zeros(20)

    # Training loop
    while epsilon_hat < epsilon:
        # Train teacher models
        teacher_models = []
        for temp_x in x_partition:
            Z_mb = np.random.uniform(-1., 1., size=[partition_data_no, z_dim])
            G_mb = generator.predict(Z_mb)

            idx = np.random.permutation(len(temp_x))
            X_mb = temp_x[idx[:partition_data_no]]

            X_comb = np.concatenate((X_mb, G_mb), axis=0)
            Y_comb = np.concatenate((np.ones([partition_data_no, ]), np.zeros([partition_data_no, ])), axis=0)

            model = LogisticRegression()
            model.fit(X_comb, Y_comb)
            teacher_models.append(model)

        # Train student
        for _ in range(n_s):
            Z_mb = np.random.uniform(-1., 1., size=[batch_size, z_dim])
            G_mb = generator.predict(Z_mb)
            Y_mb = []

            for j in range(batch_size):
                n0, n1, r_j = pate_lambda(G_mb[j], teacher_models, lamda)
                Y_mb.append(r_j)

                # Update moments accountant
                q = np.exp(np.log(2 + lamda * abs(n0 - n1)) - np.log(4.0) - (lamda * abs(n0 - n1)))

                for l in range(20):
                    temp1 = 2 * (lamda ** 2) * (l + 1) * (l + 2)
                    temp2 = (1 - q) * (((1 - q) / (1 - q * np.exp(2 * lamda))) ** (l + 1)) + q * np.exp(2 * lamda * (l + 1))
                    alpha[l] += min(temp1, np.log(temp2))

            Y_mb = np.reshape(Y_mb, [-1, 1])

            with tf.GradientTape() as tape:
                S_fake = student(G_mb, training=True)
                S_loss = tf.reduce_mean(Y_mb * S_fake) - tf.reduce_mean((1 - Y_mb) * S_fake)

            grads = tape.gradient(S_loss, student.trainable_variables)
            student_optimizer.apply_gradients(zip(grads, student.trainable_variables))

        # Train generator
        Z_mb = np.random.uniform(-1., 1., size=[batch_size, z_dim])
        with tf.GradientTape() as tape:
            G_mb = generator(Z_mb, training=True)
            S_fake = student(G_mb)
            G_loss = -tf.reduce_mean(S_fake)

        grads = tape.gradient(G_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        # Update epsilon_hat
        curr_list = [(alpha[l] + np.log(1 / delta)) / float(l + 1) for l in range(20)]
        epsilon_hat = min(curr_list)

    # Generate synthetic data
    x_train_hat = generator.predict(np.random.uniform(-1., 1., size=[no, z_dim]))
    return x_train_hat