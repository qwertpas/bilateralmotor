import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sympy import *

# Global variables to store precompiled expressions
_symbolic_eigenvalues = None
_symbolic_params = None

def _initialize_symbolic_expressions():
    global _symbolic_eigenvalues, _symbolic_params
    if _symbolic_eigenvalues is not None:
        return

    # Define symbolic variables
    x_h = symbols('x_h')
    x_j, x_j_dot, x_r, x_r_dot = symbols('x_j, \dot{x}_j, x_r, \dot{x}_r')
    m_h, m_r = symbols('m_h, m_r')
    K_pj, K_dj, K_pr, K_dr = symbols('K_pj, K_dj, K_pr, K_dr')
    _symbolic_params = (K_dj, K_dr, K_pj, K_pr, m_h, m_r)

    # Define forces
    F_r = K_pr*(x_j - x_r) + K_dr*(x_j_dot - x_r_dot)
    F_j_tr = K_pj*(x_h - x_j) + K_dj*(0 - x_j_dot)
    # F_j_fb = -F_r             #with no force scaling
    F_j_fb = -(m_h/m_r) * F_r   #with force scaling by mass ratio
    F_j_tot = F_j_tr + F_j_fb

    # Compute accelerations
    x_j_ddot = F_j_tot / m_h
    x_r_ddot = F_r / m_r

    # Build state matrix
    A = Matrix([
        [diff(x_j_ddot, x_j_dot), diff(x_j_ddot, x_j), diff(x_j_ddot, x_r_dot), diff(x_j_ddot, x_r)],
        [1, 0, 0, 0],
        [diff(x_r_ddot, x_j_dot), diff(x_r_ddot, x_j), diff(x_r_ddot, x_r_dot), diff(x_r_ddot, x_r)],
        [0, 0, 1, 0]
    ])

    # Precompute eigenvalue expressions
    _symbolic_eigenvalues = list(A.eigenvals().keys())


def compute_eigenvalues_sympy(K_dj_num, K_dr_num, K_pj_num, K_pr_num, m_h_num, m_r_num):
    # Initialize symbolic expressions if not already done
    _initialize_symbolic_expressions()
    
    # Create substitution dictionary
    subs_dict = dict(zip(_symbolic_params, 
                        (K_dj_num, K_dr_num, K_pj_num, K_pr_num, m_h_num, m_r_num)))
    
    # Evaluate eigenvalues with numerical values
    return [e.subs(subs_dict).evalf() for e in _symbolic_eigenvalues]

def plot_eigenvalues_interactive():
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.35)  # Make room for sliders
    
    # Initial parameter values
    K_dj_init = 1.0
    K_dr_init = 1.0
    K_pj_init = 1.0
    K_pr_init = 1.0
    m_h_init = 1.0
    m_r_init = 2.0
    
    # Calculate initial eigenvalues
    eigs = compute_eigenvalues_sympy(K_dj_init, K_dr_init, K_pj_init, K_pr_init, m_h_init, m_r_init)
    eigs = [complex(e) for e in eigs]  # Convert to complex numbers
    scatter = ax.scatter([e.real for e in eigs], [e.imag for e in eigs], c='red', marker='o', s=100)
    
    # Set up the plot
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title('Eigenvalues in Complex Plane')
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Create slider axes
    slider_color = 'lightgoldenrodyellow'
    ax_K_pj = plt.axes([0.1, 0.25, 0.65, 0.03], facecolor=slider_color)
    ax_K_dj = plt.axes([0.1, 0.20, 0.65, 0.03], facecolor=slider_color)
    ax_K_pr = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=slider_color)
    ax_K_dr = plt.axes([0.1, 0.10, 0.65, 0.03], facecolor=slider_color)
    ax_m_h = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=slider_color)
    ax_m_r = plt.axes([0.1, 0.00, 0.65, 0.03], facecolor=slider_color)
    
    # Create sliders
    s_K_pj = Slider(ax_K_pj, 'Kp_j', 0.0, 5.0, valinit=K_pj_init)
    s_K_dj = Slider(ax_K_dj, 'Kd_j', 0.0, 5.0, valinit=K_dj_init)
    s_K_pr = Slider(ax_K_pr, 'Kp_r', 0.0, 5.0, valinit=K_pr_init)
    s_K_dr = Slider(ax_K_dr, 'Kd_r', 0.0, 5.0, valinit=K_dr_init)
    s_m_h = Slider(ax_m_h, 'm_h', 0.1, 5.0, valinit=m_h_init)
    s_m_r = Slider(ax_m_r, 'm_r', 0.1, 5.0, valinit=m_r_init)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-2.5, 2.5)
    
    def update(val):
        # Get current slider values
        K_dj = s_K_dj.val
        K_dr = s_K_dr.val
        K_pj = s_K_pj.val
        K_pr = s_K_pr.val
        m_h = s_m_h.val
        m_r = s_m_r.val
        
        # Calculate new eigenvalues
        eigs = compute_eigenvalues_sympy(K_dj, K_dr, K_pj, K_pr, m_h, m_r)
        eigs = [complex(e) for e in eigs]
        
        # Update scatter plot
        scatter.set_offsets(np.c_[[e.real for e in eigs], [e.imag for e in eigs]])
        
        # Update plot limits if needed
        # ax.relim()
        # ax.autoscale_view()
        fig.canvas.draw_idle()
    
    # Register the update function with each slider
    s_K_dj.on_changed(update)
    s_K_dr.on_changed(update)
    s_K_pj.on_changed(update)
    s_K_pr.on_changed(update)
    s_m_h.on_changed(update)
    s_m_r.on_changed(update)
    
    plt.show()

if __name__ == "__main__":
    plot_eigenvalues_interactive()