import numpy
import tensorflow as tf


def function_factory(pinn, tf_train_u_X, tf_train_u_Y, tf_train_f_X, np_val_X, np_val_ic, val_T, tf_val_Y,
                     epochs_per_print, u_loss_weight, f_loss_weight, save_losses):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        pinn [in]: an instance of a `pinn.PINN` subclasses.
        tf_train_u_X [in]: tensorflow train inputs for MSEu.
        tf_train_u_Y [in]: tensorflow train outputs for MSEu.
        tf_train_f_X [in]: tensorflow train inputs for MSEf.
        tf_val_X [in]: tensorflow validation inputs.
        tf_val_Y [in]: tensorflow validation outputs.
        epochs_per_print [in]: the number of epochs between loss printings.
        u_loss_weight [in]: MSEu weight in total loss.
        f_loss_weight [in]: MSEf weight in total loss.
        save_losses [in]: to save or not MSE, MSEu and MSEf status in the PINN object.
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
        tf_val_NN = pinn.predict(np_val_X, np_val_ic, val_T, return_raw=True)
        tf_val_loss = tf.reduce_mean(tf.square(tf_val_NN - tf_val_Y))
        np_val_loss = tf_val_loss.numpy()
        if epoch % epochs_per_print == 0:
            print('Epoch:', str(epoch), '-', 'L-BFGS\' validation loss:', str(np_val_loss))

        # Save losses
        if save_losses:
            pinn.train_total_loss.append(tf_total_loss.numpy())
            pinn.train_u_loss.append(tf_u_loss.numpy())
            pinn.train_f_loss.append(tf_f_loss.numpy())

            pinn.validation_loss.append(np_val_loss)

        f.iter.assign_add(1)

        return tf_total_loss, grads

    # Store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters

    return f
