import torch

import matplotlib.pyplot as plt

# Number of points
num_points = 100

# Generate evenly spaced angles on the unit circle
theta = torch.linspace(0, 2 * torch.pi, num_points)

# Calculate x and y values
x = torch.cos(theta)
y = torch.sin(theta)

# Recover theta using atan2
recovered_theta = torch.atan2(y, x)

# Plot the original and recovered theta
plt.figure(figsize=(8, 6))
plt.plot(theta.numpy(), label="Original Theta")
plt.plot(recovered_theta.numpy(), "--", label="Recovered Theta (atan2)")
plt.xlabel("Index")
plt.ylabel("Theta (radians)")
plt.legend()
plt.title("Original vs Recovered Theta")
plt.show()
