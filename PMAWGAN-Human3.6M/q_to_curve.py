import numpy as np
from numpy import linalg as LA
from scipy import integrate


def q_to_curves( q ):
    [L, F] = np.shape(q)
    s = np.linspace(0, 1, F)
    qnorm = np.zeros([F, 1])
    for i in range(F):
        qnorm[i] = LA.norm(q[:, i], 2)

    s = np.expand_dims(s, axis=0)
    curve = np.zeros([L, F])
    for i in range(L):
         temp = np.multiply(q[i,:],np.transpose(qnorm))
         curve[i, :] = integrate.cumtrapz(temp, s, initial=0)
    return curve




