import numpy
import tensorflow as tf


def function_factory(pinn, tf_train_u_X, tf_train_u_Y, tf_train_f_X, tf_val_X, tf_val_Y,
                     epochs_over_analysis, u_loss_weight, f_loss_weight, attach_losses):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(model, train_x, train_y, collocation_inputs).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # Obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(pinn.model.trainable_variables)
    n_tensors = len(shapes)

    # We'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = numpy.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    # @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            pinn.model.trainable_variables[i].assign(tf.reshape(param, shape))

    # Now create a function that will be returned by this factory
    # @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor. params_1d are the weights of the network
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # Use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # Update the parameters in the model
            assign_new_model_parameters(params_1d)
            # Calculate losses
            tf_total_loss, tf_u_loss, tf_f_loss = pinn.get_losses(tf_train_u_X, tf_train_u_Y, tf_train_f_X,
                                                                  u_loss_weight, f_loss_weight)

        # Calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(tf_total_loss, pinn.model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # Validation
        epoch = f.iter.numpy()
        tf_val_NN = pinn.model(tf_val_X)
        tf_val_loss = tf.reduce_mean(tf.square(tf_val_NN - tf_val_Y))
        np_val_loss = tf_val_loss.numpy()
        if epoch % epochs_over_analysis == 0:
            print('Epoch:', str(epoch), '-', 'L-BFGS\' validation loss:', str(np_val_loss))

        # Save losses
        if attach_losses:
            pinn.train_total_loss.append(tf_total_loss.numpy())
            pinn.train_u_loss.append(tf_u_loss.numpy())
            pinn.train_f_loss.append(tf_f_loss.numpy())

            pinn.validation_loss.append(np_val_loss)

        f.iter.assign_add(1)

        return tf_total_loss, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters

    return f
