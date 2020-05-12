#! Simulator
#! =====================
from casadi import *
from numpy import *
from pylab import *

# Studying Casadi...
#! We will investigate the working of Simulator with the help of the parametrically exited Duffing equation:
#!


# t = SX.sym('t')
#
# u = SX.sym('u')
# v = SX.sym('v')
# states = vertcat(u,v)
#
# eps   = SX.sym('eps')
# mu    = SX.sym('mu')
# alpha = SX.sym('alpha')
# k     = SX.sym('k')
# sigma = SX.sym('sigma')
# Omega = 2 + eps*sigma
#
# params = vertcat(eps,mu,alpha,k,sigma)
# rhs    = vertcat(v,-u-eps*(2*mu*v+alpha*u**3+2*k*u*cos(Omega*t)))
#
# #! We will simulate over 50 seconds, 1000 timesteps.
# dae={'x':states, 'p':params, 't':t, 'ode':rhs}
# ts = linspace(0, 50, 1000)
# integrator = integrator('integrator', 'cvodes', dae, {'grid':ts, 'output_t0':True})
#
# sol = integrator(x0=[1,0], p=[0.1,0.1,0.1,0.3,0.1])
#
# #! Plot the solution
# plot(array(sol['xf'])[0,:], array(sol['xf'])[1,:])
# xlabel('u')
# ylabel('u_dot')
# show()

# Time
t = SX.sym('t')

# States
h1 = SX.sym('h1')
h2 = SX.sym('h2')
h3 = SX.sym('h3')
h4 = SX.sym('h4')

states = vertcat(h1, h2, h3, h4)

# Controls
v1 = SX.sym('v1')
v2 = SX.sym('v2')

# Params
g = SX.sym('g')

a1 = SX.sym('a1')
a2 = SX.sym('a2')
a3 = SX.sym('a3')
a4 = SX.sym('a4')

A1 = SX.sym('A1')
A2 = SX.sym('A2')
A3 = SX.sym('A3')
A4 = SX.sym('A4')

alpha1 = SX.sym('alpha1')
alpha2 = SX.sym('alpha2')

k1 = SX.sym('k1')
k2 = SX.sym('k2')

params = vertcat(g, a1, a2, a3, a4, A1, A2, A3, A4, alpha1, alpha2, k1, k2)
rhs = vertcat(-(a1/A1)*sqrt(2*g*h1) + (a3/A1)*sqrt(2*g*h3) + ((alpha1*k1)/A1)*v1,
              -(a2/A2)*sqrt(2*g*h2) + (a4/A2)*sqrt(2*g*h4) + ((alpha2*k2)/A2)*v2,
              -(a3/A3)*sqrt(2*g*h3) + (((1 - alpha2)*k2)/A3)*v2,
              -(a4/A4)*sqrt(2*g*h4) + (((1 - alpha1)*k1)/A4)*v1)
dae={'x': states, 'p': params, 't': t, 'ode': rhs}
ts = linspace(0, 50, 1000)
integrator = integrator('integrator', 'cvodes', dae, {'grid': ts, 'output_t0': True})
sol = integrator(x0=[0, 0, 0, 0], p=[981,  # g [cm/s^2]
                                     0.071,  # a1 [cm^2]
                                     0.057,  # a2 [cm^2]
                                     0.071,  # a3 [cm^2]
                                     0.057,  # a4 [cm^2]
                                     28,  # A1 [cm^2]
                                     32,  # A2 [cm^2]
                                     28,  # A3 [cm^2]
                                     32,  # A4 [cm^2]
                                     0.5,  # alpha1 [adm]
                                     0.5,  # alpha2 [adm]
                                     3.33,  # k1 [cm^3/Vs]
                                     3.35  # k2 [cm^3/Vs]
                                     ])
