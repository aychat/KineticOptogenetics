from scipy.integrate import odeint, simps, ode
import numpy as np
from Pr_ODE_solver import Pr_ODE, Pr_ODE_jacobian
from Pfr_ODE_solver import Pfr_ODE, Pfr_ODE_jacobian


def kinetic_function(parameters):

    # ===========================================================================#
    # ------------------- READ SPECTRAL PARAMETERS FROM FILE --------------------#
    # ===========================================================================#

    data = np.loadtxt("Cph8_RefSpectra.csv", delimiter=',')

    lamb = np.array(data[:, 0])
    Pr_abs = np.array(data[:, 1])
    Pr_ems = np.array(data[:, 2])
    Pfr_abs = np.array(data[:, 3])
    Pfr_ems = np.array(data[:, 4])

    # ===========================================================================#
    # -------GET ABSORPTION EMISSION DATA OF PR AND PFR STATES BY SCALING--------#
    # ===========================================================================#
    Pr_abs *= 60000 / Pr_abs.max()
    Pfr_abs *= 50000 / Pfr_abs.max()
    Pr_ems *= 60000 / Pr_ems.max()
    Pfr_ems *= 50000 / Pfr_ems.max()

    # ===========================================================================#
    # -------------- PUMP DUMP PULSE TRANSITION RATE CALCULATION  ---------------#
    # ===========================================================================#
    pump_central = 625.
    pump_bw = 40.

    dump_central = 835.
    dump_bw = 10.

    pump_energy, dump_energy, pump_width, dump_width, t0_pump, t0_dump = parameters

    pump_spectra = (1. / (np.sqrt(np.pi) * pump_bw)) * np.exp(-((lamb - pump_central) / pump_bw) ** 2)

    dump_spectra = (1. / (np.sqrt(np.pi) * dump_bw)) * np.exp(-((lamb - dump_central) / dump_bw) ** 2)

    beam_diameter = 200.
    beam_area = np.pi * (beam_diameter ** 2) / 4.

    T_max = 3.5     # Simulation time in picoseconds
    T_steps = 1000
    t_axis = np.linspace(0., T_max, T_steps)

    I_pump = np.exp(-((t_axis - t0_pump) / pump_width) ** 2)
    I_dump = np.exp(-((t_axis - t0_dump) / dump_width) ** 2)

    scale_pump = 10e6 * pump_energy / simps(I_pump, t_axis)
    scale_dump = 10e6 * dump_energy / simps(I_dump, t_axis)

    pump_spectra *= scale_pump
    dump_spectra *= scale_dump

    K_Pr_12_pump = simps(1.92e3 * pump_spectra * lamb * Pr_abs / beam_area, lamb)
    K_Pr_12_dump = simps(1.92e3 * dump_spectra * lamb * Pr_abs / beam_area, lamb)

    K_Pr_34_pump = simps(1.92e3 * pump_spectra * lamb * Pr_ems / beam_area, lamb)
    K_Pr_34_dump = simps(1.92e3 * dump_spectra * lamb * Pr_ems / beam_area, lamb)

    K_Pfr_12_pump = simps(1.92e3 * pump_spectra * lamb * Pfr_abs / beam_area, lamb)
    K_Pfr_12_dump = simps(1.92e3 * dump_spectra * lamb * Pfr_abs / beam_area, lamb)

    K_Pfr_34_pump = simps(1.92e3 * pump_spectra * lamb * Pfr_ems / beam_area, lamb)
    K_Pfr_34_dump = simps(1.92e3 * dump_spectra * lamb * Pfr_ems / beam_area, lamb)

    A_41_Pr = 1 / .150
    A_23_Pr = 1 / .150
    A_35_Pr = 1 / 68.5
    A_34_Pr = A_35_Pr * 2.5

    A_41_Pfr = 1 / .050
    A_23_Pfr = 1 / .050
    A_35_Pfr = 1 / 2.5
    A_34_Pfr = A_35_Pfr * 5.0

    Iterations = 51

    PR = np.zeros([Iterations, 5])
    PFR = np.zeros_like(PR)

    PR[0][0] = PFR[0][0] = 0.5

    for i in range(Iterations-1):
        PR[i][0] += PFR[i][4]
        PFR[i][0] += PR[i][4]
        PR[i][4] = 0.0
        PFR[i][4] = 0.0
        sol_Pr = odeint(Pr_ODE, PR[i], t_axis,
                        args=(K_Pr_12_pump, K_Pr_12_dump, K_Pr_34_pump, K_Pr_34_dump,
                              t0_pump, pump_width, t0_dump, dump_width),
                        Dfun=Pr_ODE_jacobian)
        sol_Pfr = odeint(Pfr_ODE, PFR[i], t_axis,
                         args=(K_Pfr_12_pump, K_Pfr_12_dump, K_Pfr_34_pump, K_Pfr_34_dump,
                               t0_pump, pump_width, t0_dump, dump_width),
                         Dfun=Pfr_ODE_jacobian)

        PR[i+1][:] = sol_Pr[-1]
        PFR[i + 1][:] = sol_Pfr[-1]

    return PR[-1].sum()
