from multiprocessing import Pool
import numpy as np
from kinetics_prop import KineticsProp
import timeit

if __name__ == '__main__':

    start = timeit.default_timer()

    from itertools import product
    import pickle

    ##############################################################################
    #                                                                            #
    #                               DEFINING GRIDS                               #
    #                                                                            #
    ##############################################################################

    N1 = N2 = N3 = N4 = N5 = N6 = 3

    pump_energy = np.linspace(0.1, 0.4, N1)
    dump_energy = np.linspace(1.0, 3.0, N2)
    pump_width = np.linspace(.050, .150, N3)
    dump_width = np.linspace(.100, .200, N4)
    t0_pump = np.linspace(0.25, 0.75, N5)
    t0_dump = np.linspace(0.30, 0.80, N6)

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

        T_max=100.0,
        T_steps=1000,

        A_41=1 / .150,
        A_23=1 / .150,
        A_35=1 / 68.5,
        A_34=2.5 / 68.5,
        A_56=1 / 0.1,

        A_96=1 / .050,
        A_78=1 / .050,
        A_810=1 / 2.5,
        A_89=2.0,
        A_101=1 / 0.1,

        Iterations=51,
    )

    result = Pool(4).map(
        KineticsProp(**kinetic_params),
        product(pump_energy, dump_energy, pump_width, dump_width, t0_pump, t0_dump)
    )

    print result

    result = np.array(result).reshape(N1, N2, N3, N4, N5, N6)
    time = timeit.default_timer() - start

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
                'time': time
            },
            file_out
        )
