
#!/usr/bin/env python3
"""
Plot trajectory data from trajectory_export.json
"""
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Load trajectory data
with open('trajectory_output.json', 'r') as f:
    data = json.load(f)

samples = data['samples']

# Extract data arrays
indices = [s['index'] for s in samples]
t_s = [s['time'] for s in samples]
lat = [s['lat'] for s in samples]
lon = [s['lon'] for s in samples]
alt = [s['altitude'] for s in samples]
velocity = [s['velocity'] for s in samples]
x_m = [s['x'] for s in samples]
y_m = [s['y'] for s in samples]

# Create figure with subplots
fig = plt.figure(figsize=(15, 8))

# 1. Altitude vs Time
ax1 = plt.subplot(2, 2, 1)
ax1.plot(t_s, alt, 'b-', linewidth=2)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Altitude [m]')
ax1.set_title('Altitude Profile')
ax1.grid(True, alpha=0.3)

# 2. Velocity vs Time
ax2 = plt.subplot(2, 2, 2)
ax2.plot(t_s, velocity, 'r-', linewidth=2)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Velocity [m/s]')
ax2.set_title('Velocity Profile')
ax2.grid(True, alpha=0.3)

# 3. Lat/Lon trajectory (map view)
ax3 = plt.subplot(2, 2, 3)
ax3.plot(lon, lat, 'g-', linewidth=2, marker='o', markersize=3, markevery=max(1, len(lon)//20))
ax3.plot(lon[0], lat[0], 'go', markersize=10, label='Start')
ax3.plot(lon[-1], lat[-1], 'ro', markersize=10, label='End')
ax3.set_xlabel('Longitude [deg]')
ax3.set_ylabel('Latitude [deg]')
ax3.set_title('Ground Track')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.axis('equal')

# 4. XY trajectory with altitude color coding
ax4 = plt.subplot(2, 2, 4)
scatter = ax4.scatter([x/1000 for x in x_m], [y/1000 for y in y_m], c=alt, cmap='viridis', s=10)
ax4.plot([x_m[0]/1000], [y_m[0]/1000], 'go', markersize=10, label='Start')
ax4.plot([x_m[-1]/1000], [y_m[-1]/1000], 'ro', markersize=10, label='End')
plt.colorbar(scatter, ax=ax4, label='Altitude [m]')
ax4.set_xlabel('X [km]')
ax4.set_ylabel('Y [km]')
ax4.set_title('Horizontal Path (colored by altitude)')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.axis('equal')

plt.tight_layout()
plt.savefig('trajectory_plots.png', dpi=150, bbox_inches='tight')
print(f"Saved plots to trajectory_plots.png")
print(f"Summary:")
print(f"  Duration: {t_s[-1]:.1f} s ({t_s[-1]/60:.1f} min)")
print(f"  Max altitude: {max(alt):.1f} m")
print(f"  Max velocity: {max(velocity):.1f} m/s")
print(f"  Samples: {len(samples)}")
