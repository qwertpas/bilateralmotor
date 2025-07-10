import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches
import warnings
import time
import queue
import threading
warnings.filterwarnings('ignore')

class InteractiveHapticPlot:
    def __init__(self):
        # Initialize parameters
        self.k_r = 14000     # Robot spring (2D pair 1)
        self.b_r = 500       # Robot damping
        self.k_h = 14000     # Human spring (2D pair 2)
        self.b_h = 500       # Human damping
        self.delay_r2h = 10  # Robot to human delay (2D pair 3)
        self.delay_h2r = 10  # Human to robot delay
        self.C_fb = 1.0      # Feedback coupling (1D slider)
        
        # Fixed parameters
        self.m_h = 3.5
        self.m_r = 8.75
        self.dt = 0.001
        self.dt_stability_check = 0.001
        self.sim_time = 1
        self.step_amplitude = 0.100
        self.stability_threshold = 0.003
        
        # Force saturation limits
        self.F_robot_max = 200  # Maximum robot force (N)
        self.F_human_max = 200   # Maximum human force (N)
        
        # Parameter ranges
        self.spring_range = (1000, 30000)
        self.damping_range = (100, 2000)
        self.delay_range = (1, 50)
        self.C_fb_range = (0.0, 1.5)
        
        # Queue for thread communication
        self.update_queue = queue.Queue()
        
        # Flags for cancelling background calculations
        self.calculation_flags = {
            'robot': threading.Event(),
            'human': threading.Event(),
            'delays': threading.Event(),
            'coupling': threading.Event()
        }
        
        self.setup_figure()
        self.update_status("Initializing sliders...", 'yellow')
        self.create_sliders()
        self.update_status("Generating initial plot...", 'yellow')
        self.update_plot()
        self.update_status("Ready", 'lightgreen')
        
    def profile_time(self, func_name):
        """Decorator for timing functions"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                print(f"{func_name}: {end - start:.3f}s")
                return result
            return wrapper
        return decorator
        
    def setup_figure(self):
        """Create figure layout"""
        self.fig = plt.figure(figsize=(15, 8))
        
        # Main plot
        self.ax_main = plt.subplot2grid((4, 6), (0, 0), colspan=6, rowspan=2)
        self.ax_main.set_title('Bilateral Haptic System - Step Response', fontsize=14, fontweight='bold')
        self.ax_main.set_xlabel('Time (s)')
        self.ax_main.set_ylabel('Position (mm)')
        self.ax_main.grid(True, alpha=0.3)
        
        # Add status indicator - Make it more visible
        self.status_text = self.ax_main.text(0.95, 0.95, "Ready", 
                                           transform=self.ax_main.transAxes, 
                                           fontsize=12, fontweight='bold',
                                           horizontalalignment='right', 
                                           verticalalignment='top',
                                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', 
                                                   alpha=0.9, edgecolor='black', linewidth=1))
        
        # 2D sliders
        self.ax_robot = plt.subplot2grid((4, 6), (2, 0), colspan=2)
        self.ax_human = plt.subplot2grid((4, 6), (2, 2), colspan=2) 
        self.ax_delays = plt.subplot2grid((4, 6), (2, 4), colspan=2)
        
        # 1D slider
        self.ax_coupling = plt.subplot2grid((4, 6), (3, 1), colspan=4)
        
        for ax, title in zip([self.ax_robot, self.ax_human, self.ax_delays], 
                           ['Robot (k_r, b_r)', 'Human (k_h, b_h)', 'Delays (r2h, h2r)']):
            ax.set_title(title, fontsize=12)
            ax.set_aspect('equal')
            
    def create_sliders(self):
        """Create all sliders"""
        self.setup_2d_slider(self.ax_robot, self.spring_range, self.damping_range, 
                           self.k_r, self.b_r, 'robot')
        self.setup_2d_slider(self.ax_human, self.spring_range, self.damping_range,
                           self.k_h, self.b_h, 'human') 
        self.setup_2d_slider(self.ax_delays, self.delay_range, self.delay_range,
                           self.delay_r2h, self.delay_h2r, 'delays')
        
        self.coupling_slider = Slider(self.ax_coupling, 'Coupling (C_fb)', 
                                    self.C_fb_range[0], self.C_fb_range[1], 
                                    valinit=self.C_fb, valstep=0.01)
        self.coupling_slider.on_changed(self.update_coupling)
        
    def setup_2d_slider(self, ax, x_range, y_range, x_val, y_val, slider_type):
        """Setup 2D slider with stability background"""
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        # Create background - REDUCED RESOLUTION FOR SPEED
        resolution = 10  # Reduced from 15 (100 vs 225 simulations)
        x_bg = np.linspace(x_range[0], x_range[1], resolution)
        y_bg = np.linspace(y_range[0], y_range[1], resolution)
        X_bg, Y_bg = np.meshgrid(x_bg, y_bg)
        
        position_map = self.calculate_position_map(X_bg, Y_bg, slider_type)
        
        # Create custom colormap with black for unstable regions
        cmap = plt.cm.rainbow.copy()
        cmap.set_under('black')  # Set color for values below vmin
        
        im = ax.imshow(position_map, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
                      origin='lower', cmap=cmap, aspect='auto', vmin=50, vmax=200)
        plt.colorbar(im, ax=ax, label='Max Position (mm)', extend='min')
        
        marker = ax.plot(x_val, y_val, 'ko', markersize=8, markerfacecolor='white',
                        markeredgecolor='black', markeredgewidth=2)[0]
        
        setattr(self, f'{slider_type}_bg', im)
        setattr(self, f'{slider_type}_marker', marker)
        
        ax.figure.canvas.mpl_connect('button_press_event', 
                                   lambda event: self.on_2d_click(event, ax, slider_type))
        
    def calculate_position_map(self, X, Y, slider_type):
        """Calculate maximum position for each background point"""
        position_map = np.zeros(X.shape)
        
        total_points = X.shape[0] * X.shape[1]
        total_processed = 0
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Set temporary parameters based on slider type
                temp_params = {
                    'k_r': self.k_r, 'b_r': self.b_r, 'k_h': self.k_h, 'b_h': self.b_h,
                    'delay_r2h': self.delay_r2h, 'delay_h2r': self.delay_h2r, 'C_fb': self.C_fb
                }
                
                if slider_type == 'robot':
                    temp_params['k_r'], temp_params['b_r'] = X[i, j], Y[i, j]
                elif slider_type == 'human':
                    temp_params['k_h'], temp_params['b_h'] = X[i, j], Y[i, j]
                elif slider_type == 'delays':
                    temp_params['delay_r2h'], temp_params['delay_h2r'] = int(X[i, j]), int(Y[i, j])
                
                # Simulate system to get maximum position
                max_pos = self.simulate_for_max_position(**temp_params)
                position_map[i, j] = max_pos
                
                total_processed += 1
                
        return position_map
        
    def get_color_for_position(self, max_pos):
        """Get color based on maximum position value"""
        if max_pos < 0:  # Unstable
            return 'black'
        else:
            # Normalize color based on position (50 to 200mm range for color scaling)
            norm_pos = np.clip((max_pos - 50) / (200 - 50), 0, 1)
            return plt.cm.rainbow(norm_pos)

    def simulate_for_max_position(self, k_r, b_r, k_h, b_h, delay_r2h, delay_h2r, C_fb):
        """Simulate system and return maximum absolute position or special value for unstable"""
        try:
            sim_steps = int(self.sim_time / self.dt_stability_check)
            max_delay = max(int(delay_r2h), int(delay_h2r))
            
            x_r = np.zeros(sim_steps)
            x_r_dot = np.zeros(sim_steps)
            x_h = np.zeros(sim_steps)
            x_h_dot = np.zeros(sim_steps)
            F_robot = np.zeros(sim_steps)
            F_human = np.zeros(sim_steps)
            
            step_start = int(0.1 / self.dt_stability_check)
            reference = np.zeros(sim_steps)
            reference[step_start:] = self.step_amplitude
            
            for i in range(max_delay, sim_steps):
                x_h_delayed = x_h[i-int(delay_r2h)] if i >= delay_r2h else 0
                x_h_dot_delayed = x_h_dot[i-int(delay_r2h)] if i >= delay_r2h else 0
                F_robot_delayed = F_robot[i-int(delay_h2r)] if i >= delay_h2r else 0
                
                F_robot[i] = k_r*(x_h_delayed - x_r[i-1]) + b_r*(x_h_dot_delayed - x_r_dot[i-1])
                F_human[i] = k_h*(reference[i] - x_h[i-1]) + b_h*(0 - x_h_dot[i-1]) - C_fb * F_robot_delayed
                
                F_robot[i] = np.clip(F_robot[i], -self.F_robot_max, self.F_robot_max)
                F_human[i] = np.clip(F_human[i], -self.F_human_max, self.F_human_max)
                
                x_h_ddot = F_human[i] / self.m_h
                x_r_ddot = F_robot[i] / self.m_r
                
                x_h_dot[i] = x_h_dot[i-1] + x_h_ddot * self.dt_stability_check
                x_r_dot[i] = x_r_dot[i-1] + x_r_ddot * self.dt_stability_check
                
                x_h[i] = x_h[i-1] + x_h_dot[i] * self.dt_stability_check
                x_r[i] = x_r[i-1] + x_r_dot[i] * self.dt_stability_check
            
            # Check stability in last 0.1s
            final_samples = int(0.1 / self.dt_stability_check)  # Number of samples in final 0.1s
            final_stable = all(abs(x_r[-final_samples:] - x_h[-final_samples:]) < self.stability_threshold)
            
            if not final_stable:
                return -1  # Special value for unstable regions
            
            # Return maximum absolute position in millimeters for stable regions
            max_pos = max(np.max(np.abs(x_r)), np.max(np.abs(x_h))) * 1000
            return max_pos
            
        except Exception as e:
            return -1  # Return special value for error cases
            
    def update_status(self, message, color='lightblue'):
        """Update the status indicator"""
        self.status_text.set_text(message)
        bbox = self.status_text.get_bbox_patch()
        bbox.set_facecolor(color)
        self.fig.canvas.draw()  # Force immediate update
        plt.pause(0.001)  # Brief pause to ensure display
        
    def on_2d_click(self, event, ax, slider_type):
        """Handle 2D slider clicks"""
        if event.inaxes == ax and event.xdata is not None:
            # Cancel any ongoing calculations for this slider
            self.calculation_flags[slider_type].set()
            
            # Show immediate feedback
            self.update_status(f"Clicked {slider_type} slider - Updating...", 'orange')
            
            x_click, y_click = event.xdata, event.ydata
            
            if slider_type == 'robot':
                self.k_r, self.b_r = x_click, y_click
            elif slider_type == 'human':
                self.k_h, self.b_h = x_click, y_click
            elif slider_type == 'delays':
                self.delay_r2h, self.delay_h2r = int(x_click), int(y_click)
            
            # Update marker
            getattr(self, f'{slider_type}_marker').set_data([x_click], [y_click])
            
            # Update main plot first
            self.update_status("Updating step response plot...", 'yellow')
            self.update_plot()
            
            # Then update backgrounds with status
            self.update_status("Generating maps...", 'red')
            start_time = time.time()
            self.update_all_backgrounds()
            end_time = time.time()
            print(f"Position maps generated in {end_time - start_time:.2f}s")
            
            # Complete
            self.update_status("Ready", 'lightgreen')
                
    def update_coupling(self, val):
        """Update coupling parameter"""
        # Cancel any ongoing calculations
        for flag in self.calculation_flags.values():
            flag.set()
        
        # Show immediate feedback
        self.update_status("Coupling changed - Updating...", 'orange')
        
        self.C_fb = val
        
        self.update_status("Generating maps...", 'red')
        self.update_all_backgrounds()
        
        self.update_status("Updating plot...", 'yellow')
        self.update_plot()
        
        self.update_status("Ready", 'lightgreen')
        
    def update_slider_background(self, slider_type, ax, x_range, y_range):
        """Update a single slider's background in a separate thread"""
        resolution = 10
        x_bg = np.linspace(x_range[0], x_range[1], resolution)
        y_bg = np.linspace(y_range[0], y_range[1], resolution)
        X_bg, Y_bg = np.meshgrid(x_bg, y_bg)
        
        # Position calculation
        position_map = self.calculate_position_map(X_bg, Y_bg, slider_type)
        
        # Image update
        bg_image = getattr(self, f'{slider_type}_bg')
        bg_image.set_array(position_map)
        self.fig.canvas.draw_idle()
        
    def calculate_background_thread(self, slider_type, x_range, y_range):
        """Calculate background map in a separate thread"""
        # Clear any existing flag and set new one
        self.calculation_flags[slider_type].clear()
        current_flag = self.calculation_flags[slider_type]
        
        resolution = 10
        x_bg = np.linspace(x_range[0], x_range[1], resolution)
        y_bg = np.linspace(y_range[0], y_range[1], resolution)
        X_bg, Y_bg = np.meshgrid(x_bg, y_bg)
        
        # Position calculation
        position_map = np.zeros(X_bg.shape)
        for i in range(X_bg.shape[0]):
            # Check if calculation has been cancelled
            if current_flag.is_set():
                return
                
            for j in range(X_bg.shape[1]):
                temp_params = {
                    'k_r': self.k_r, 'b_r': self.b_r, 'k_h': self.k_h, 'b_h': self.b_h,
                    'delay_r2h': self.delay_r2h, 'delay_h2r': self.delay_h2r, 'C_fb': self.C_fb
                }
                
                if slider_type == 'robot':
                    temp_params['k_r'], temp_params['b_r'] = X_bg[i, j], Y_bg[i, j]
                elif slider_type == 'human':
                    temp_params['k_h'], temp_params['b_h'] = X_bg[i, j], Y_bg[i, j]
                elif slider_type == 'delays':
                    temp_params['delay_r2h'], temp_params['delay_h2r'] = int(X_bg[i, j]), int(Y_bg[i, j])
                
                position_map[i, j] = self.simulate_for_max_position(**temp_params)
                
                # Check if calculation has been cancelled
                if current_flag.is_set():
                    return
        
        # Only put result in queue if calculation wasn't cancelled
        if not current_flag.is_set():
            self.update_queue.put((slider_type, position_map))
        
    def calculate_1d_background_thread(self):
        """Calculate 1D coupling background in a separate thread"""
        # Clear any existing flag and set new one
        self.calculation_flags['coupling'].clear()
        current_flag = self.calculation_flags['coupling']
        
        C_values = np.linspace(self.C_fb_range[0], self.C_fb_range[1], 10)
        results = []
        
        for C in C_values[:-1]:
            # Check if calculation has been cancelled
            if current_flag.is_set():
                return
                
            max_pos = self.simulate_for_max_position(self.k_r, self.b_r, self.k_h, self.b_h, 
                                    self.delay_r2h, self.delay_h2r, C)
            results.append((C, max_pos))
        
        # Only put result in queue if calculation wasn't cancelled
        if not current_flag.is_set():
            self.update_queue.put(('coupling', results))
            
    def update_all_backgrounds(self):
        """Update all slider backgrounds with detailed profiling"""
        configs = [('robot', self.ax_robot, self.spring_range, self.damping_range),
                  ('human', self.ax_human, self.spring_range, self.damping_range),
                  ('delays', self.ax_delays, self.delay_range, self.delay_range)]
        
        # Cancel any ongoing calculations
        for flag in self.calculation_flags.values():
            flag.set()
        
        # Start new background threads
        threads = []
        for slider_type, ax, x_range, y_range in configs:
            thread = threading.Thread(target=self.calculate_background_thread,
                                   args=(slider_type, x_range, y_range))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Start coupling background thread
        coupling_thread = threading.Thread(target=self.calculate_1d_background_thread)
        coupling_thread.daemon = True
        coupling_thread.start()
        threads.append(coupling_thread)
        
        # Process results as they come in
        results_received = 0
        while results_received < len(threads):
            try:
                slider_type, result = self.update_queue.get(timeout=0.1)
                
                if slider_type == 'coupling':
                    # Update coupling slider background
                    for patch in self.ax_coupling.patches[:]:
                        if hasattr(patch, '_stability_patch'):
                            patch.remove()
                    
                    # Calculate width based on value spacing
                    C_values = np.linspace(self.C_fb_range[0], self.C_fb_range[1], 10)
                    rect_width = C_values[1] - C_values[0]
                    
                    for C, max_pos in result:
                        color = self.get_color_for_position(max_pos)
                        rect = patches.Rectangle((C, -0.5), 
                                              rect_width, 1.0,
                                              facecolor=color, alpha=0.9, zorder=0)
                        rect._stability_patch = True
                        self.ax_coupling.add_patch(rect)
                else:
                    # Update 2D slider background
                    bg_image = getattr(self, f'{slider_type}_bg')
                    bg_image.set_array(result)
                
                results_received += 1
                self.fig.canvas.draw()
                
            except queue.Empty:
                # Check if any thread is still alive
                if not any(t.is_alive() for t in threads):
                    break
        
        # Clean up any remaining threads
        for thread in threads:
            thread.join(timeout=0.1)

    def update_1d_background(self):
        """Update 1D slider background"""
        # Clear patches
        for patch in self.ax_coupling.patches[:]:
            if hasattr(patch, '_stability_patch'):
                patch.remove()
        
        # Generate values
        C_values = np.linspace(self.C_fb_range[0], self.C_fb_range[1], 10)
        
        # Position calculations
        for i, C in enumerate(C_values[:-1]):
            max_pos = self.simulate_for_max_position(self.k_r, self.b_r, self.k_h, self.b_h, 
                                     self.delay_r2h, self.delay_h2r, C)
            
            if max_pos < 0:  # Unstable
                color = 'black'
                alpha = 0.9
            else:
                # Normalize color based on position (50 to 200mm range for color scaling)
                norm_pos = np.clip((max_pos - 50) / (200 - 50), 0, 1)
                color = plt.cm.rainbow(norm_pos)
                alpha = 0.9
            
            rect = patches.Rectangle((C, -0.5), C_values[1] - C_values[0], 1.0,
                                   facecolor=color, alpha=alpha, zorder=0)
            rect._stability_patch = True
            self.ax_coupling.add_patch(rect)
        
    def update_plot(self):
        """Update main step response plot"""
        try:
            # Simulate system
            sim_steps = int(self.sim_time / self.dt)
            time_vec = np.arange(0, self.sim_time, self.dt)
            max_delay = max(int(self.delay_r2h), int(self.delay_h2r))
            
            x_r = np.zeros(sim_steps)
            x_r_dot = np.zeros(sim_steps)
            x_h = np.zeros(sim_steps)
            x_h_dot = np.zeros(sim_steps)
            F_robot = np.zeros(sim_steps)
            F_human = np.zeros(sim_steps)
            
            step_start = int(0.1 / self.dt)
            reference = np.zeros(sim_steps)
            reference[step_start:] = self.step_amplitude
            
            for i in range(max_delay, sim_steps):
                x_h_delayed = x_h[i-int(self.delay_r2h)] if i >= self.delay_r2h else 0
                x_h_dot_delayed = x_h_dot[i-int(self.delay_r2h)] if i >= self.delay_r2h else 0
                F_robot_delayed = F_robot[i-int(self.delay_h2r)] if i >= self.delay_h2r else 0
                
                F_robot[i] = self.k_r*(x_h_delayed - x_r[i-1]) + self.b_r*(x_h_dot_delayed - x_r_dot[i-1])
                F_human[i] = self.k_h*(reference[i] - x_h[i-1]) + self.b_h*(0 - x_h_dot[i-1]) - self.C_fb * F_robot_delayed
                
                F_robot[i] = np.clip(F_robot[i], -self.F_robot_max, self.F_robot_max)
                F_human[i] = np.clip(F_human[i], -self.F_human_max, self.F_human_max)
                
                x_h_ddot = F_human[i] / self.m_h
                x_r_ddot = F_robot[i] / self.m_r
                
                x_h_dot[i] = x_h_dot[i-1] + x_h_ddot * self.dt
                x_r_dot[i] = x_r_dot[i-1] + x_r_ddot * self.dt
                
                x_h[i] = x_h[i-1] + x_h_dot[i] * self.dt
                x_r[i] = x_r[i-1] + x_r_dot[i] * self.dt
            
            # Store current status text and color before clearing
            if hasattr(self, 'status_text'):
                current_status = self.status_text.get_text()
                current_color = self.status_text.get_bbox_patch().get_facecolor()
            else:
                current_status = "Ready"
                current_color = 'lightgreen'
            
            # Plot
            self.ax_main.clear()
            self.ax_main.plot(time_vec, x_r*1000, 'b-', linewidth=2, label='Robot Position')
            self.ax_main.plot(time_vec, x_h*1000, 'r-', linewidth=2, label='Human Position')
            self.ax_main.plot(time_vec, reference*1000, 'k--', linewidth=1, label='Reference')
            self.ax_main.set_title('Bilateral Haptic System - Step Response', fontsize=14, fontweight='bold')
            self.ax_main.set_xlabel('Time (s)')
            self.ax_main.set_ylabel('Position (mm)')
            self.ax_main.grid(True, alpha=0.3)
            self.ax_main.legend()
            
            # # Stability indicator - use full simulation for current parameters only
            # is_stable = self.is_stable_full_sim(self.k_r, self.b_r, self.k_h, self.b_h, 
            #                                   self.delay_r2h, self.delay_h2r, self.C_fb)
            # status = "STABLE" if is_stable else "UNSTABLE"
            # color = 'green' if is_stable else 'red'
            # self.ax_main.text(0.02, 0.98, f"System: {status}", 
            #                 transform=self.ax_main.transAxes, fontsize=12, fontweight='bold',
            #                 verticalalignment='top', color=color,
            #                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Recreate status text with previous status
            self.status_text = self.ax_main.text(0.95, 0.95, current_status,
                                               transform=self.ax_main.transAxes,
                                               fontsize=12, fontweight='bold',
                                               horizontalalignment='right',
                                               verticalalignment='top',
                                               bbox=dict(boxstyle='round,pad=0.5',
                                                       facecolor=current_color,
                                                       alpha=0.9, edgecolor='black',
                                                       linewidth=1))
            
        except Exception as e:
            self.ax_main.clear()
            self.ax_main.text(0.5, 0.5, f"Error: {str(e)}", transform=self.ax_main.transAxes,
                            ha='center', va='center', fontsize=12, color='red')
            
        self.fig.canvas.draw()

if __name__ == "__main__":
    plt.ion()
    app = InteractiveHapticPlot()
    plt.tight_layout()
    plt.show()
    input("Press Enter to close...")
