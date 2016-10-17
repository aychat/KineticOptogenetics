import numpy as np


def Pr_ODE(y_Pr, t, K_Pr_12_pump, K_Pr_12_dump, K_Pr_34_pump, K_Pr_34_dump, t0_pump, pump_width,
                                             t0_dump, dump_width):

    Intensity_pump = np.exp(-np.power(((t- t0_pump) / pump_width), 2))
    Intensity_dump = np.exp(-np.power(((t - t0_dump) / dump_width), 2))

    A_41 = 1/.150
    A_23 = 1/.150
    A_35 = 1/68.5
    A_34 = A_35 * 2.5
    A_53 = 0.0

    K_Pr_12_pump /= 1e12
    K_Pr_12_dump /= 1e12
    K_Pr_34_pump /= 1e12
    K_Pr_34_dump /= 1e12

    C_12 = Intensity_pump * K_Pr_12_pump + Intensity_dump * K_Pr_12_dump
    C_34 = Intensity_pump * K_Pr_34_pump + Intensity_dump * K_Pr_34_dump

    R1_dot = C_12 * (y_Pr[1] - y_Pr[0]) + A_41 * y_Pr[3]
    R2_dot = C_12 * (y_Pr[0] - y_Pr[1]) - A_23 * y_Pr[1]
    R3_dot = C_34 * (y_Pr[3] - y_Pr[2]) + A_23 * y_Pr[1] - A_34 * y_Pr[2] - A_35 * y_Pr[2]
    R4_dot = C_34 * (y_Pr[2] - y_Pr[3]) - A_41 * y_Pr[3] + A_34 * y_Pr[2]
    R5_dot = A_35 * y_Pr[2]

    return [R1_dot, R2_dot, R3_dot, R4_dot, R5_dot]

def Pr_ODE_jacobian(y_Pr, t, K_Pr_12_pump, K_Pr_12_dump, K_Pr_34_pump, K_Pr_34_dump, t0_pump, pump_width,
                                             t0_dump, dump_width):

    Intensity_pump = np.exp(-np.power(((t - t0_pump) / pump_width), 2))
    Intensity_dump = np.exp(-np.power(((t - t0_dump) / dump_width), 2))

    A_41 = 1/.150
    A_23 = 1/.150
    A_35 = 1/68.5
    A_34 = A_35 * 2.5
    A_53 = 0.0

    C_12 = Intensity_pump * K_Pr_12_pump + Intensity_dump * K_Pr_12_dump
    C_34 = Intensity_pump * K_Pr_34_pump + Intensity_dump * K_Pr_34_dump

    return [[-C_12, C_12, 0, A_41, 0], [C_12, -C_12-A_23, 0, 0, 0], [0, A_23, -A_34-C_34-A_35, C_34, 0],
            [0, 0, A_34+C_34, -C_34-A_41, 0], [0, 0, A_35, 0, 0]]

def Pr_ODE_check(y_Pr, t, t0_pump, pump_width, t0_dump, dump_width):

    Intensity_pump = np.exp(-np.power(((t - t0_pump) / pump_width), 2))
    Intensity_dump = np.exp(-np.power(((t - t0_dump) / dump_width), 2))

    A_41 = 1/.150
    A_23 = 1/.150
    A_35 = 1/68.5
    A_34 = A_35 * 2.5
    A_53 = 0.0

    K_Pr_12_pump = 15.1 #32243948564500.1
    K_Pr_12_dump = 0.0
    K_Pr_34_pump = 15.1 #909537215679.152
    K_Pr_34_dump = 0.0

    dt = t[1] - t[0]
    R1_dot = (Intensity_pump * K_Pr_12_pump + Intensity_dump * K_Pr_12_dump) * (y_Pr[:,1] - y_Pr[:,0]) \
             + A_41 * y_Pr[:,3]
    R2_dot = (Intensity_pump * K_Pr_12_pump + Intensity_dump * K_Pr_12_dump) * (y_Pr[:,0] - y_Pr[:,1]) -\
             A_23 * y_Pr[:,1]
    R3_dot = (Intensity_pump * K_Pr_34_pump + Intensity_dump * K_Pr_34_dump) * (y_Pr[:,3] - y_Pr[:,2]) + \
             A_23 * y_Pr[:,1] - A_34 * y_Pr[:,2] - A_35 * y_Pr[:,2]
    R4_dot = (Intensity_pump * K_Pr_34_pump + Intensity_dump * K_Pr_34_dump) * (y_Pr[:,2] - y_Pr[:,3]) - \
             A_41 * y_Pr[:,3] + A_34 * y_Pr[:,2]
    R5_dot = A_35 * y_Pr[:,2]

    return np.array([
        R1_dot-np.gradient(y_Pr[:,0], dt),
        R2_dot-np.gradient(y_Pr[:,1], dt),
        R3_dot-np.gradient(y_Pr[:,2], dt),
        R4_dot-np.gradient(y_Pr[:,3], dt),
        R5_dot-np.gradient(y_Pr[:,4], dt)
    ])