from multiprocessing import Pool
import numpy as np
from Pr_ODE_solver import Pr_ODE, Pr_ODE_jacobian
from Pfr_ODE_solver import Pfr_ODE, Pfr_ODE_jacobian
from kinetics_prop import KineticsProp

if __name__ == '__main__':

    from itertools import product
    import pickle

    ##############################################################################
    #                                                                            #
    #                               DEFINING GRIDS                               #
    #                                                                            #
    ##############################################################################

    N1 = N2 = N3 = N4 = N5 = N6 = 1

    pump_energy = np.linspace(0.25, 0.25, N1)
    dump_energy = np.linspace(2.0, 2.0, N2)
    pump_width = np.linspace(.100, .100, N3)
    dump_width = np.linspace(.150, .150, N4)
    t0_pump = np.linspace(0.5, 0.5, N5)
    t0_dump = np.linspace(0.5251, .5251, N6)

    ##############################################################################
    #                                                                            #
    #                           DEFINING KINETIC PARAMETERS                      #
    #                                                                            #
    ##############################################################################

    kinetic_params = dict(
        # Pulses characterization
        pump_central=625.,
        pump_bw=40.,

        dump_central=835.,
        dump_bw=10.,

        beam_diameter=200.,

        T_max=3.5,
        T_steps=1000,

        A_41=1 / .150,
        A_23=1 / .150,
        A_35=1 / 68.5,
        A_34=2.5 / 68.5,
        A_56=1 / 100.,

        A_96=1 / .050,
        A_78=1 / .050,
        A_810=1 / 2.5,
        A_89=2.0,
        A_101=1 / 100.,

        Iterations=51,
    )

    result = Pool(4).map(
        KineticsProp(**kinetic_params),
        product(pump_energy, dump_energy, pump_width, dump_width, t0_pump, t0_dump)
    )

    print result

    result = np.array(result).reshape(N1, N2, N3, N4, N5, N6)

    with open('result.pickle', 'wb') as file_out:
        pickle.dump(
            {
                'kinetic_params' : kinetic_params,
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