import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# --- Cycloid Function ---
def compute_cycloid(R, theta_vals):
    x_vals = R * (theta_vals - np.sin(theta_vals))
    y_vals = -R * (1 - np.cos(theta_vals))
    return x_vals, y_vals

# --- Initial Values ---
R_init = 1.0
T_total_init = 2.0
theta_vals = np.linspace(0, 2 * np.pi, 300)
x_vals, y_vals = compute_cycloid(R_init, theta_vals)

# --- Set up Figure and Axes ---
fig, (ax_curve, ax_energy) = plt.subplots(2, 1, figsize=(10, 10))
plt.subplots_adjust(left=0.1, bottom=0.3, hspace=0.5)

# Cycloid Plot
line, = ax_curve.plot(x_vals, y_vals, 'k-', label='Cycloid')
dot, = ax_curve.plot([], [], 'ro', markersize=8)
time_text = ax_curve.text(0.05, 0.9, '', transform=ax_curve.transAxes, fontsize=12)

ax_curve.set_xlim(0, np.max(x_vals) + 0.5)
ax_curve.set_ylim(np.min(y_vals) - 0.5, 0.5)
ax_curve.set_aspect('equal')
ax_curve.set_xlabel('x [m]')
ax_curve.set_ylabel('y [m]')
ax_curve.set_title('Cycloid: Brachistochrone & Tautochrone')
ax_curve.legend()

# Energy Plot
energy_times = []
potential_energies = []
kinetic_energies = []

potential_line, = ax_energy.plot([], [], 'b-', label='Potential Energy')
kinetic_line, = ax_energy.plot([], [], 'r-', label='Kinetic Energy')
total_line, = ax_energy.plot([], [], 'g--', label='Total Energy')

ax_energy.set_xlim(0, T_total_init)
ax_energy.set_ylim(0, 1.2)
ax_energy.set_xlabel('Time [s]')
ax_energy.set_ylabel('Energy (normalized)')
ax_energy.set_title('Energy Over Time')
ax_energy.legend()

# --- Widgets ---
ax_slider_radius = plt.axes([0.25, 0.2, 0.65, 0.03])
slider_radius = Slider(ax_slider_radius, 'Radius (R)', 0.1, 5.0, valinit=R_init)

ax_slider_time = plt.axes([0.25, 0.15, 0.65, 0.03])
slider_time = Slider(ax_slider_time, 'Total Time (s)', 0.5, 5.0, valinit=T_total_init)

ax_button = plt.axes([0.45, 0.05, 0.1, 0.05])
button = Button(ax_button, 'Pause', color='lightblue', hovercolor='skyblue')

# --- Variables ---
is_paused = False
frame_idx = 0
R = R_init
T_total = T_total_init

# --- Functions ---
def init():
    dot.set_data([], [])
    time_text.set_text('')
    potential_line.set_data([], [])
    kinetic_line.set_data([], [])
    total_line.set_data([], [])
    energy_times.clear()
    potential_energies.clear()
    kinetic_energies.clear()
    return dot, time_text, potential_line, kinetic_line, total_line

def animate(i):
    global frame_idx
    if is_paused:
        i = frame_idx
    else:
        frame_idx = i % len(x_vals)
    
    if len(x_vals) == 0 or frame_idx >= len(x_vals):
        return dot, time_text, potential_line, kinetic_line, total_line
    
    dot.set_data([x_vals[frame_idx]], [y_vals[frame_idx]])
    
    t = frame_idx / (len(x_vals) - 1) * T_total
    time_text.set_text(f'Time = {t:.2f} s')
    
    update_energy(frame_idx, t)
    return dot, time_text, potential_line, kinetic_line, total_line

def update_energy(idx, current_time):
    if len(y_vals) == 0:
        return
    y = y_vals[idx]
    h = max(-y_vals[0], 1e-6)  # protect against division by 0
    current_h = -y
    potential = np.clip(current_h / h, 0, 1)
    kinetic = np.clip(1 - potential, 0, 1)
    
    energy_times.append(current_time)
    potential_energies.append(potential)
    kinetic_energies.append(kinetic)
    
    potential_line.set_data(energy_times, potential_energies)
    kinetic_line.set_data(energy_times, kinetic_energies)
    total_line.set_data(energy_times, np.ones_like(energy_times))  # total energy stays 1

    ax_energy.set_xlim(0, max(T_total, current_time + 0.5))  # Extend x-axis if needed

def update(val):
    global R, x_vals, y_vals, frame_idx
    R = slider_radius.val
    T_total = slider_time.val
    x_vals, y_vals = compute_cycloid(R, theta_vals)
    line.set_data(x_vals, y_vals)
    
    ax_curve.set_xlim(0, np.max(x_vals) + 0.5)
    ax_curve.set_ylim(np.min(y_vals) - 0.5, 0.5)
    
    frame_idx = 0
    init()  # restart energies
    fig.canvas.draw_idle()

def toggle_pause(event):
    global is_paused
    is_paused = not is_paused
    button.label.set_text('Play' if is_paused else 'Pause')

slider_radius.on_changed(update)
slider_time.on_changed(update)
button.on_clicked(toggle_pause)

# --- Animation ---
ani = FuncAnimation(fig, animate, frames=len(theta_vals), init_func=init,
                    interval=20, blit=False, repeat=True)

plt.show()
