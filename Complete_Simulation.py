import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Physical parameters (natural units: c = ℏ = 1)
v = 246e9 # Higgs VEV (eV)
rho_Planck = 5e96 # Planck density (kg/m³)
kappa_range = np.linspace(0.1, 1.0, 5) # VA coupling range
xi = 0.1 # Curvature coupling

# Higgs effective mass function
def m_H(rho):
return v * np.sqrt(1 - np.exp(-(rho / rho_Planck)**2))

# Higgs-VA differential equation
def higgs_va(y, t, kappa, R_curv):
phi, dphi_dt = y
rho = 1e80 * np.exp(-t**2) # Density profile (example)
m_eff = m_H(rho)
d2phi_dt2 = - (m_eff**2 + xi * R_curv + kappa * rho**2) * phi
return [dphi_dt, d2phi_dt2]

# Initial conditions and time
t = np.linspace(0, 10, 1000)
phi0 = v # Vacuum expectation value (ET)
dphi_dt0 = 0.0 # Zero initial derivative

# Simulation for different kappa
plt.figure(figsize=(10, 6))
for kappa in kappa_range:
R_curv = 1e-50 * (1 + np.tanh(t - 5)) # Variable curvature (example)
sol = odeint(higgs_va, [phi0, dphi_dt0], t, args=(kappa, R_curv)) 
plt.plot(t, sol[:, 0], label=f'κ = {kappa:.1f}')

# Display
plt.title('Evolution of the Higgs in the Absolute Vacuum', fontsize=14)
plt.xlabel('Time (dimensionalless)', fontsize=12)
plt.ylabel('Higgs field (Φ)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
