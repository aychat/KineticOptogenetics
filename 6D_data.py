from multiprocessing import Pool
import numpy as np
from Pr_ODE_solver import Pr_ODE, Pr_ODE_jacobian
from Pfr_ODE_solver import Pfr_ODE, Pfr_ODE_jacobian
from Zak_kinetics_function import kinetic_function
import time


if __name__ == '__main__':

    from itertools import product
    import pickle

    p = Pool(4)

    N1 = N2 = N3 = N4 = N5 = N6 = 1

    pump_energy = np.linspace(0.25, 0.25, N1)
    dump_energy = np.linspace(2.0, 2.0, N2)
    pump_width = np.linspace(.100, .100, N3)
    dump_width = np.linspace(.150, .150, N4)
    t0_pump = np.linspace(0.5, 0.5, N5)
    t0_dump = np.linspace(0.5251, .5251, N6)

    result = p.map(kinetic_function, product(pump_energy, dump_energy, pump_width, dump_width, t0_pump, t0_dump))

    print result

    result = np.array(result).reshape(N1, N2, N3, N4, N5, N6)

    with open('result.pickle', 'wb') as file_out:
        pickle.dump(
            {
                'params': {
                    'pump_energy': pump_energy,
                    'dump_energy': dump_energy,
                    'pump_width': pump_width,
                    'dump_width': dump_width,
                    't0_pump': t0_pump,
                    't0_dump': t0_dump
                },
                'result': result,
            },
            file_out
        )

    #print(p.map(f, [1, 2, 3]))