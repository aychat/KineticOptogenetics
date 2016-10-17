import numpy as np

def Pfr_ODE(y_Pfr, t, K_Pfr_12_pump, K_Pfr_12_dump, K_Pfr_34_pump, K_Pfr_34_dump, t0_pump, pump_width,
                                             t0_dump, dump_width):

    Intensity_pump = np.exp(-np.power(((t - t0_pump) / pump_width), 2))
    Intensity_dump = np.exp(-np.power(((t - t0_dump) / dump_width), 2))

    A_41 = 1/.050
    A_23 = 1/.050
    A_35 = 1/2.5
    A_34 = A_35 * 5.0
    A_53 = 0.0

    K_Pfr_12_pump /= 1e12
    K_Pfr_12_dump /= 1e12
    K_Pfr_34_pump /= 1e12
    K_Pfr_34_dump /= 1e12

    C_12 = Intensity_pump * K_Pfr_12_pump + Intensity_dump * K_Pfr_12_dump
    C_34 = Intensity_pump * K_Pfr_34_pump + Intensity_dump * K_Pfr_34_dump

    FR1_dot = C_12 * (y_Pfr[1] - y_Pfr[0]) + A_41 * y_Pfr[3]
    FR2_dot = C_12 * (y_Pfr[0] - y_Pfr[1]) - A_23 * y_Pfr[1]
    FR3_dot = C_34 * (y_Pfr[3] - y_Pfr[2]) + A_23 * y_Pfr[1] - A_34 * y_Pfr[2] - A_35 * y_Pfr[2]
    FR4_dot = C_34 * (y_Pfr[2] - y_Pfr[3]) - A_41 * y_Pfr[3] + A_34 * y_Pfr[2]
    FR5_dot = A_35 * y_Pfr[2]

    return [FR1_dot, FR2_dot, FR3_dot, FR4_dot, FR5_dot]

def Pfr_ODE_jacobian(y_Pfr, t, K_Pfr_12_pump, K_Pfr_12_dump, K_Pfr_34_pump, K_Pfr_34_dump, t0_pump, pump_width,
                                             t0_dump, dump_width):

    Intensity_pump = np.exp(-np.power(((t - t0_pump) / pump_width), 2))
    Intensity_dump = np.exp(-np.power(((t - t0_dump) / dump_width), 2))

    A_41 = 1/.150
    A_23 = 1/.150
    A_35 = 1/68.5
    A_34 = A_35 * 2.5
    A_53 = 0.0

    C_12 = Intensity_pump * K_Pfr_12_pump + Intensity_dump * K_Pfr_12_dump
    C_34 = Intensity_pump * K_Pfr_34_pump + Intensity_dump * K_Pfr_34_dump

    return [[-C_12, C_12, 0, A_41, 0], [C_12, -C_12-A_23, 0, 0, 0], [0, A_23, -A_34-C_34-A_35, C_34, 0],
            [0, 0, A_34+C_34, -C_34-A_41, 0], [0, 0, A_35, 0, 0]]