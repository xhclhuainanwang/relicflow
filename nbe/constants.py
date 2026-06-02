import math

MIN_NUMBER = 1.0e-100
X_BEXP = 100.0

# DRAKE auxiliary.wl constants
M_PL_REDUCED = 1.2209e19 / math.sqrt(8.0 * math.pi)
KELVIN_GEV = 8.617343e-14
METER_1_PER_GEV = 1.0 / (197.326961e-18)
T_TODAY = 2.725 * KELVIN_GEV
RHO_CRITICAL = 1.05368e-5 / (0.01 * METER_1_PER_GEV) ** 3

