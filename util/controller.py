import casadi as cs
import numpy as np


class PINNController:
    def __init__(self, pinn_model, system_simulator):
        self.weights = pinn_model.get_weights()[1:]
        self.X_normalizer = pinn_model.X_normalizer
        self.Y_normalizer = pinn_model.Y_normalizer

        self.system_simulator = system_simulator

        self.optimizer = cs.Opti()

    def nn(self, cs_x):
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

    def predict_horizon(self, last_u, np_ref, np_y0, np_min_u, np_max_u, np_min_y, np_max_y, prediction_horizon, T,
                        outputs_to_control=None, use_runge_kutta=False):
        n = int(prediction_horizon / T)

        cs_u = self.optimizer.variable(n, np_min_u.shape[1])
        cs_y = self.optimizer.variable(n, np_y0.shape[1])

        self.optimizer.set_initial(cs_u, np.tile(np_max_u, (cs_u.shape[0], 1)))
        self.optimizer.set_initial(cs_y, np.tile(np_y0, (cs_y.shape[0], 1)))

        if use_runge_kutta:
            runge_kutta = self.system_simulator.get_runge_kutta(T)
            cs_nn = runge_kutta(u=cs.horzcat(*[cs_u[0, j] for j in range(cs_u.shape[1])]), y0=np_y0)['yf']
        else:
            cs_x = cs.horzcat(T,
                              *[cs_u[0, j] for j in range(cs_u.shape[1])],
                              *[np_y0[0, j] for j in range(np_y0.shape[1])])
            cs_nn = self.nn(cs_x)

        for j in range(max(cs_nn.shape[1], cs_u.shape[1])):
            if j < cs_nn.shape[1]:
                self.optimizer.subject_to(cs_nn[0, j] == cs_y[0, j])
                self.optimizer.subject_to(cs_y[0, j] >= np_min_y[0, j])
                self.optimizer.subject_to(cs_y[0, j] <= np_max_y[0, j])
            if j < cs_u.shape[1]:
                self.optimizer.subject_to(cs_u[0, j] >= np_min_u[0, j])
                self.optimizer.subject_to(cs_u[0, j] <= np_max_u[0, j])
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
        
        
        cost_function = cs.MX.sym('J') #simuolic representation of the cost function
        L_sym = cs.MX.sym('L') #simbolic representation of the lagrangian
        cost_ode = {'x':cost_function,'p':L_sym,'ode':L_sym} #integrator ode to compute cost function
        J = 0 #Cost function in optiomization variable terms
        if outputs_to_control is None:
            for i in range(cs_y.shape[0]):
                L = sum((cs_y[i, :] - np_ref[i, :]) ** 2) #Add terms to the lagrangian
                cost_F = cs.integrator('F','cvodes',cost_ode,{'t0':0,'tf':T}) #implement cost function, integrated
                J = cost_F(x0=J,p = L) #add result to optimization.

        else:
            # Peso 10 para o erro em relação ao controle
            L = 0
            for i in range(cs_y.shape[0]):
                for j in outputs_to_control:
                    L = L + 10*(cs_y[i, j] - np_ref[i, j]) ** 2
                    if i == 0:
                        L = L + (cs_u[i , j] - last_u[j]) ** 2
                    if i >= 1:
                        L = L + (cs_u[i , j] - cs_u[i-1 , j]) ** 2
                cost_F = cs.integrator('F','cvodes',cost_ode,{'t0':0,'tf':T})
                result = cost_F(x0=J,p = L)
                J = J + result['xf']

        self.optimizer.minimize(J)
        self.optimizer.solver('ipopt')
        sol = self.optimizer.solve()
        return sol.value(cs_u)

    def control(self, np_ref, np_y0, np_min_u, np_max_u, np_min_y, np_max_y,
                sim_time, prediction_horizon, T, collocation_points_per_T,
                outputs_to_control=None, use_runge_kutta=False):
        self.optimizer = cs.Opti()

        n = int(prediction_horizon / T)

        np_T = np.linspace(0, T, collocation_points_per_T)
        np_t = np_T

        np_states = np_y0
        us = np.array([0, 0, 0, 0])
        us = self.predict_horizon(us, np_ref[:n, :], np_y0, np_min_u, np_max_u, np_min_y, np_max_y, prediction_horizon, T,
                                  outputs_to_control, use_runge_kutta)

        np_controls = np.tile(us[0], (collocation_points_per_T, np_min_u.shape[1]))
        np_new_ref = np.tile(np_ref[0, :], (collocation_points_per_T, 1))
        np_states = np.append(np_states,
                              self.system_simulator.run(np_T, np.array(us[0]), np_states[0], output_t0=False),
                              axis=0)

        for i in range(1, int(sim_time / T)):
            np_t = np.append(np_t, np_T[1:] + np_t[-1])
            np_ic = np.reshape(np_states[-1], np_y0.shape)

            if i + n > np_ref.shape[0]:
                np_ref_window = np_ref[-n:, :]
            else:
                np_ref_window = np_ref[i:i + n, :]

            us = self.predict_horizon(us[0], np_ref_window, np_ic, np_min_u, np_max_u, np_min_y, np_max_y,
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
