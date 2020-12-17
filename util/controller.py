import casadi as cs
import numpy as np


class PINNController:
    """Generates control examples based on a given PINN model, a system simulator and some control parameters."""

    def __init__(self, pinn_model, system_simulator):
        """
        Loads PINN's parameters to use them in a Casadi function that replicates the neural network's behavior. It also
        saves a reference to the system simulator to estimate its response due to the control solution.

        :param pinn_model: PINN trained model
        :type pinn_model: util.pinn.PINN
        :param system_simulator: Real system simulator
        :type system_simulator: util.systems.*
        """

        self.weights = pinn_model.get_weights()[1:]
        self.X_normalizer = pinn_model.X_normalizer
        self.Y_normalizer = pinn_model.Y_normalizer

        self.system_simulator = system_simulator

        self.optimizer = cs.Opti()

    def nn(self, cs_x):
        """
        A Casadi replica of the neural network. A tahn(tanh(...)) function.

        :param cs_x: Input vector (t, u, nn_0)
        :type cs_x: casadi.MX
        :returns: Neural network's outputs
        :rtype: casadi.MX
        """

        cs_a = self.X_normalizer.normalize(cs_x)
        L = len(self.weights) - 1
        for l in range(L):
            np_W = self.weights[l][0]
            np_b = np.reshape(self.weights[l][1], (1, self.weights[l][1].shape[0]))
            cs_a = cs.tanh(cs_a @ np_W + np_b)
        np_W = self.weights[L][0]
        np_b = np.reshape(self.weights[L][1], (1, self.weights[L][1].shape[0]))
        cs_a = cs_a @ np_W + np_b

        return self.Y_normalizer.denormalize(cs_a)

    def predict_horizon(self, np_ref, np_y0, np_min_u, np_max_u, np_min_y, np_max_y, prediction_horizon, T,
                        outputs_to_control=None, use_runge_kutta=False):
        """
        Uses a PINN or a Runge-Kutta model to optimize the control signals for a prediction horizon basing on a
        reference and some control and output constraints.

        :param np_ref: Reference in each optimization point.
        :type np_ref: numpy.ndarray
        :param np_y0: Sampled initial conditions.
        :type np_y0: numpy.ndarray
        :param np_min_u: Min control constraint.
        :type np_min_u: numpy.ndarray
        :param np_max_u: Max control constraint.
        :type np_max_u: numpy.ndarray
        :param np_min_y: Min output constraint.
        :type np_min_y: numpy.ndarray
        :param np_max_y: Max output constraint.
        :type np_max_y: numpy.ndarray
        :param prediction_horizon: Prediction horizon, multiple of T.
        :type prediction_horizon: float
        :param T: Controller's sample period.
        :type T: float
        :param outputs_to_control: Which outputs must follow the reference. If it is None, consider all.
            (default is None)
        :type outputs_to_control: list
        :param use_runge_kutta: Sets the function to use Runge-Kutta model instead of PINN model.
            (default is False)
        :type use_runge_kutta: bool
        :returns: Optimized control signal for each T.
        :rtype: numpy.ndarray
        """

        # Number of optimization points
        n = int(prediction_horizon / T)

        # Optimizer's variables
        cs_u = self.optimizer.variable(n, np_min_u.shape[1])
        cs_y = self.optimizer.variable(n, np_y0.shape[1])

        # Initial conditions
        self.optimizer.set_initial(cs_u, np.tile(np_max_u, (cs_u.shape[0], 1)))
        self.optimizer.set_initial(cs_y, np.tile(np_y0, (cs_y.shape[0], 1)))

        # Calculus of the first state after the first control input
        if use_runge_kutta:
            runge_kutta = self.system_simulator.get_runge_kutta(T)
            cs_nn = runge_kutta(u=cs.horzcat(*[cs_u[0, j] for j in range(cs_u.shape[1])]), y0=np_y0)['yf']
        else:
            cs_x = cs.horzcat(T,
                              *[cs_u[0, j] for j in range(cs_u.shape[1])],
                              *[np_y0[0, j] for j in range(np_y0.shape[1])])
            cs_nn = self.nn(cs_x)

        # Constraints for the first step
        for j in range(max(cs_nn.shape[1], cs_u.shape[1])):
            if j < cs_nn.shape[1]:
                # Binding of the neural network's output to the optimizer's variable
                self.optimizer.subject_to(cs_nn[0, j] == cs_y[0, j])

                # Minimum and maximum output constraints
                self.optimizer.subject_to(cs_y[0, j] >= np_min_y[0, j])
                self.optimizer.subject_to(cs_y[0, j] <= np_max_y[0, j])
            if j < cs_u.shape[1]:
                # Minimum and maximum input constraints
                self.optimizer.subject_to(cs_u[0, j] >= np_min_u[0, j])
                self.optimizer.subject_to(cs_u[0, j] <= np_max_u[0, j])

        # Output calculus and constraint settings for the subsequent steps (same procedure)
        for i in range(1, n):
            if use_runge_kutta:
                cs_nn = runge_kutta(u=cs.horzcat(*[cs_u[i, j] for j in range(cs_u.shape[1])]),
                                    y0=cs.horzcat(*[cs_y[i - 1, j] for j in range(cs_y.shape[1])]))['yf']
            else:
                cs_x = cs.horzcat(T,
                                  *[cs_u[i, j] for j in range(cs_u.shape[1])],
                                  *[cs_y[i - 1, j] for j in range(cs_y.shape[1])])
                cs_nn = self.nn(cs_x)

            for j in range(max(cs_nn.shape[1], cs_u.shape[1])):
                if j < cs_nn.shape[1]:
                    self.optimizer.subject_to(cs_nn[0, j] == cs_y[i, j])
                    self.optimizer.subject_to(cs_y[i, j] >= np_min_y[0, j])
                    self.optimizer.subject_to(cs_y[i, j] <= np_max_y[0, j])
                if j < cs_u.shape[1]:
                    self.optimizer.subject_to(cs_u[i, j] >= np_min_u[0, j])
                    self.optimizer.subject_to(cs_u[i, j] <= np_max_u[0, j])

        # Cost function
        if outputs_to_control is None:
            cost_function = sum((cs_y[i, j] - np_ref[i, j]) ** 2
                                for j in range(cs_y.shape[1]) for i in range(cs_y.shape[0]))
        else:
            cost_function = sum((cs_y[i, j] - np_ref[i, j]) ** 2
                                for j in outputs_to_control for i in range(cs_y.shape[0]))

        # Optimization
        self.optimizer.minimize(cost_function)
        self.optimizer.solver('ipopt')
        sol = self.optimizer.solve()

        return sol.value(cs_u)

    def control(self, np_ref, np_y0, np_min_u, np_max_u, np_min_y, np_max_y,
                sim_time, prediction_horizon, T, collocation_points_per_T,
                outputs_to_control=None, use_runge_kutta=False):
        """
        Generates a complete control example based on the given parameters using moving horizon MPC with a PINN or a
        Runge-Kutta model.

        :param np_ref: Reference in each optimization point of the whole simulation.
        :type np_ref: numpy.ndarray
        :param np_y0: Initial conditions.
        :type np_y0: numpy.ndarray
        :param np_min_u: Min control constraint.
        :type np_min_u: numpy.ndarray
        :param np_max_u: Max control constraint.
        :type np_max_u: numpy.ndarray
        :param np_min_y: Min output constraint.
        :type np_min_y: numpy.ndarray
        :param np_max_y: Max output constraint.
        :type np_max_y: numpy.ndarray
        :param sim_time: Simulation time.
        :type sim_time: float
        :param prediction_horizon: Prediction horizon, multiple of T.
        :type prediction_horizon: float
        :param T: Controller's sample period.
        :type T: float
        :param collocation_points_per_T: The number of collocation points simulated in each T to analyse the system's
            behavior between control changes.
        :type collocation_points_per_T: int
        :param outputs_to_control: Which outputs must follow the reference. If it is None, consider all.
            (default is None)
        :type outputs_to_control: list
        :param use_runge_kutta: Sets the function to use Runge-Kutta model instead of PINN model.
            (default is False)
        :type use_runge_kutta: bool
        :returns: Time vector, control matrix, reference matrix and state matrix.
        :rtype: list of numpy.ndarray
        """

        # Optimizer cleaning
        self.optimizer = cs.Opti()

        # Number of optimization points
        n = int(prediction_horizon / T)

        # Period and time vectors
        np_T = np.linspace(0, T, collocation_points_per_T)
        np_t = np_T

        # State matrix
        np_states = np_y0

        # Control optimization for the first prediction horizon
        us = self.predict_horizon(np_ref[:n, :], np_y0, np_min_u, np_max_u, np_min_y, np_max_y, prediction_horizon, T,
                                  outputs_to_control, use_runge_kutta)

        # Control matrix
        np_controls = np.tile(us[0], (collocation_points_per_T, np_min_u.shape[1]))

        # Reference matrix
        np_new_ref = np.tile(np_ref[0, :], (collocation_points_per_T, 1))

        # Real system simulation using the first optimized control step as input
        np_states = np.append(np_states,
                              self.system_simulator.run(np_T, np.array(us[0]), np_states[0], output_t0=False),
                              axis=0)

        # Same procedure for the subsequent horizons
        for i in range(1, int(sim_time / T)):
            np_t = np.append(np_t, np_T[1:] + np_t[-1])
            np_ic = np.reshape(np_states[-1], np_y0.shape)

            if i + n > np_ref.shape[0]:
                np_ref_window = np_ref[-n:, :]
            else:
                np_ref_window = np_ref[i:i + n, :]

            us = self.predict_horizon(np_ref_window, np_ic, np_min_u, np_max_u, np_min_y, np_max_y,
                                      prediction_horizon, T, outputs_to_control, use_runge_kutta)
            np_controls = np.append(np_controls,
                                    np.tile(us[0], (collocation_points_per_T - 1, np_min_u.shape[1])),
                                    axis=0)
            np_new_ref = np.append(np_new_ref,
                                   np.tile(np_ref[i, :], (collocation_points_per_T - 1, 1)),
                                   axis=0)
            np_states = np.append(np_states,
                                  self.system_simulator.run(np_T, np.array(us[0]), np_states[-1, :], output_t0=False),
                                  axis=0)

        return np_t, np_controls, np_new_ref, np_states
