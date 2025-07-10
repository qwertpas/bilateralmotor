# Test comment
# ... existing code ...

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Constants
m = 1.0  # mass
b = 1.0  # damping
k = 1.0  # spring constant
d_c = 0.1  # distance from center of mass to muscle attachment point
d_l = 0.2  # distance from center of mass to load attachment point
I = m * (d_c**2 + d_l**2) / 12  # moment of inertia (simplified)

# The following piecewise functions for K_dj and K_dr are derived from the symbolic analysis in `bilateral_motor_sim_sym.py`
# K_dj = 
# Piecewise((K_p - K_d*K_l/C_d, C_d > 0), (K_p, True))
# K_dr = 
# Piecewise((K_d*C_l/C_d, C_d > 0), (K_d, True))

def compute_eigenvalues(K_p, K_d, K_l, C_d, C_l, tau):
    """
    Computes eigenvalues using piecewise functions for K_dj and K_dr.
    """
    if C_d > 0:
        K_dj = K_p - K_d * K_l / C_d
        K_dr = K_d * C_l / C_d
    else:
        K_dj = K_p
        K_dr = K_d
        
    A = np.array([
        [0, 1],
        [-k/I - 2*K_dj/I, -b/I - 2*K_dr/I]
    ])
    eig_vals = np.linalg.eigvals(A)
    return eig_vals

def update(val):
    K_p = slider_K_p.val
    K_d = slider_K_d.val
    K_l = slider_K_l.val
    C_d = slider_C_d.val
    C_l = slider_C_l.val
    tau = slider_tau.val
    
    eig_vals = compute_eigenvalues(K_p, K_d, K_l, C_d, C_l, tau)
    
    ax.clear()
    ax.plot(np.real(eig_vals), np.imag(eig_vals), 'o')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Eigenvalues of the System')
    ax.grid(True)
    fig.canvas.draw_idle()

if __name__ == '__main__':
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)

    axcolor = 'lightgoldenrodyellow'
    ax_K_p = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_K_d = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
    ax_K_l = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_C_d = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_C_l = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    ax_tau = plt.axes([0.25, 0.0, 0.65, 0.03], facecolor=axcolor)

    slider_K_p = Slider(ax_K_p, 'K_p', 0.1, 10.0, valinit=1)
    slider_K_d = Slider(ax_K_d, 'K_d', 0.1, 10.0, valinit=1)
    slider_K_l = Slider(ax_K_l, 'K_l', 0.1, 10.0, valinit=1)
    slider_C_d = Slider(ax_C_d, 'C_d', 0.1, 10.0, valinit=1)
    slider_C_l = Slider(ax_C_l, 'C_l', 0.1, 10.0, valinit=1)
    slider_tau = Slider(ax_tau, 'tau', 0.01, 1.0, valinit=0.1)

    slider_K_p.on_changed(update)
    slider_K_d.on_changed(update)
    slider_K_l.on_changed(update)
    slider_C_d.on_changed(update)
    slider_C_l.on_changed(update)
    slider_tau.on_changed(update)

    # Initial plot
    update(None)

    plt.show()