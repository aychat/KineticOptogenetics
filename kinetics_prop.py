from types import MethodType, FunctionType
from scipy.integrate import ode, odeint, simps
from scipy import linalg
import numpy as np
from Pr_ODE_solver import Pr_ODE, Pr_ODE_jacobian
from Pfr_ODE_solver import Pfr_ODE, Pfr_ODE_jacobian
import matplotlib.pyplot as plt

class KineticsProp:
    """
    Propagation of kinetic equations
    """

    def __init__(self, **kwargs):
        """
         The following parameters are to be specified as arguments:

               pump_central -- central wavelength of pump pulse in nm
               pump_bw -- pump bandwidth in nm
               dump_central -- central wavelength of dump pulse in nm
               dump_bw -- dump bandwidth in nm
        """

        # Save all attributes
        for name, value in kwargs.items():
            # If the value supplied is a function,
            # then dynamically assign it as a method,
            # otherwise bind it a property.
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        # ===========================================================================#
        # ------------------- READ SPECTRAL PARAMETERS FROM FILE --------------------#
        # ===========================================================================#

        data = np.loadtxt("Cph8_RefSpectra.csv", delimiter=',')

        self.lamb = np.array(data[:, 0])
        self.Pr_abs = np.array(data[:, 1])
        self.Pr_ems = np.array(data[:, 3])
        self.Pfr_abs = np.array(data[:, 2])
        self.Pfr_ems = np.array(data[:, 4])

        # ===========================================================================#
        # -------GET ABSORPTION EMISSION DATA OF PR AND PFR STATES BY SCALING--------#
        # ===========================================================================#
        self.Pr_abs *= 60000 / self.Pr_abs.max()
        self.Pfr_abs *= 50000 / self.Pfr_abs.max()
        self.Pr_ems *= 60000 / self.Pr_ems.max()
        self.Pfr_ems *= 50000 / self.Pfr_ems.max()

        self.beam_area = np.pi * self.beam_diameter ** 2 / 4.

    def I_pump(self, t):
        return np.exp(-((t - self.t0_pump) / self.pump_width) ** 2)

    def I_dump(self, t):
        return np.exp(-((t - self.t0_dump) / self.dump_width) ** 2)

    def propagate(self, G0, V_pump, V_dump, p0):
        """
        Propagate a state with withe initial condition p0
        """

        def jac(p, t):
            """
            Return Jacobian of the rate equations
            """
            tmp = V_pump * self.I_pump(t)
            tmp += V_dump * self.I_dump(t)
            tmp += G0
            return tmp

        def rhs(p, t):
            """
            Return the r.h.s. of the rate equations
            """
            return jac(p, t).dot(p)

        #prop = ode(rhs, jac)
        #prop.set_initial_value(p0, 0.0)
        #return prop.integrate(self.T_max)
        return odeint(rhs, p0, self.t_axis, Dfun=jac)

    def __call__(self, parameters):
        pump_energy, dump_energy, self.pump_width, self.dump_width, self.t0_pump, self.t0_dump = parameters

        pump_spectra = 1. / (np.sqrt(np.pi) * self.pump_bw) \
            * np.exp(-((self.lamb - self.pump_central) / self.pump_bw) ** 2)

        dump_spectra = 1. / (np.sqrt(np.pi) * self.dump_bw) \
            * np.exp(-((self.lamb - self.dump_central) / self.dump_bw) ** 2)

        self.t_axis = t_axis = np.linspace(0., self.T_max, self.T_steps)

        I_pump = self.I_pump(t_axis)
        I_dump = self.I_dump(t_axis)

        scale_pump = 10e6 * pump_energy / simps(I_pump, t_axis) # ASK ZAK
        scale_dump = 10e6 * dump_energy / simps(I_dump, t_axis)

        pump_spectra *= scale_pump
        dump_spectra *= scale_dump

        K_Pr_12_pump = simps(1.92e-9 * pump_spectra * self.lamb * self.Pr_abs / self.beam_area, self.lamb)
        K_Pr_12_dump = simps(1.92e-9 * dump_spectra * self.lamb * self.Pr_abs / self.beam_area, self.lamb)

        K_Pr_34_pump = simps(1.92e-9 * pump_spectra * self.lamb * self.Pr_ems / self.beam_area, self.lamb)
        K_Pr_34_dump = simps(1.92e-9 * dump_spectra * self.lamb * self.Pr_ems / self.beam_area, self.lamb)

        K_Pfr_12_pump = simps(1.92e-9 * pump_spectra * self.lamb * self.Pfr_abs / self.beam_area, self.lamb)
        K_Pfr_12_dump = simps(1.92e-9 * dump_spectra * self.lamb * self.Pfr_abs / self.beam_area, self.lamb)

        K_Pfr_34_pump = simps(1.92e-9 * pump_spectra * self.lamb * self.Pfr_ems / self.beam_area, self.lamb)
        K_Pfr_34_dump = simps(1.92e-9 * dump_spectra * self.lamb * self.Pfr_ems / self.beam_area, self.lamb)


        ###############################################################################

        self.G0_Pr = np.zeros([5, 5])

        self.G0_Pr[0, 3] = self.A_41_Pr
        self.G0_Pr[1, 1] = -self.A_23_Pr
        self.G0_Pr[2, 1] = self.A_23_Pr
        self.G0_Pr[2, 2] = -self.A_34_Pr-self.A_35_Pr
        self.G0_Pr[3, 2] = self.A_34_Pr
        self.G0_Pr[3, 3] = -self.A_41_Pr
        self.G0_Pr[4, 2] = self.A_35_Pr

        self.G0_Pfr = np.zeros_like(self.G0_Pr)

        self.G0_Pfr[0, 3] = self.A_41_Pfr
        self.G0_Pfr[1, 1] = -self.A_23_Pfr
        self.G0_Pfr[2, 1] = self.A_23_Pfr
        self.G0_Pfr[2, 2] = -self.A_34_Pfr - self.A_35_Pfr
        self.G0_Pfr[3, 2] = self.A_34_Pfr
        self.G0_Pfr[3, 3] = -self.A_41_Pfr
        self.G0_Pfr[4, 2] = self.A_35_Pfr

        self.V_Pr_pump = np.zeros_like(self.G0_Pr)

        self.V_Pr_pump[0, 1] = self.V_Pr_pump[1, 0] = K_Pr_12_pump
        self.V_Pr_pump[0, 0] = self.V_Pr_pump[1, 1] = -K_Pr_12_pump
        self.V_Pr_pump[2, 2] = self.V_Pr_pump[3, 3] = -K_Pr_34_pump
        self.V_Pr_pump[2, 3] = self.V_Pr_pump[3, 2] = K_Pr_34_pump

        self.V_Pr_dump = np.zeros_like(self.G0_Pr)

        self.V_Pr_dump[0, 1] = self.V_Pr_dump[1, 0] = K_Pr_12_dump
        self.V_Pr_dump[0, 0] = self.V_Pr_dump[1, 1] = -K_Pr_12_dump
        self.V_Pr_dump[2, 2] = self.V_Pr_dump[3, 3] = -K_Pr_34_dump
        self.V_Pr_dump[2, 3] = self.V_Pr_dump[3, 2] = K_Pr_34_dump

        self.V_Pfr_pump = np.zeros_like(self.G0_Pr)

        self.V_Pfr_pump[0, 1] = self.V_Pfr_pump[1, 0] = K_Pfr_12_pump
        self.V_Pfr_pump[0, 0] = self.V_Pfr_pump[1, 1] = -K_Pfr_12_pump
        self.V_Pfr_pump[2, 2] = self.V_Pfr_pump[3, 3] = -K_Pfr_34_pump
        self.V_Pfr_pump[2, 3] = self.V_Pfr_pump[3, 2] = K_Pfr_34_pump

        self.V_Pfr_dump = np.zeros_like(self.G0_Pfr)

        self.V_Pfr_dump[0, 1] = self.V_Pfr_dump[1, 0] = K_Pfr_12_dump
        self.V_Pfr_dump[0, 0] = self.V_Pfr_dump[1, 1] = -K_Pfr_12_dump
        self.V_Pfr_dump[2, 2] = self.V_Pfr_dump[3, 3] = -K_Pfr_34_dump
        self.V_Pfr_dump[2, 3] = self.V_Pfr_dump[3, 2] = K_Pfr_34_dump

        ###############################################################################
        #
        #   Transfer matrix construction
        #
        ###############################################################################
        M_Pr = np.transpose(
            [self.propagate(self.G0_Pr, self.V_Pr_pump, self.V_Pr_dump, e)[-1] for e in np.eye(5)]
        )

        M_Pfr = np.transpose(
            [self.propagate(self.G0_Pfr, self.V_Pfr_pump, self.V_Pfr_dump, e)[-1] for e in np.eye(5)]
        )

        M = linalg.block_diag(M_Pr, M_Pfr)
        M[4, 5] = M[4, 4]
        M[4, 4] = 0
        M[9, 0] = M[9, 9]
        M[9, 9] = 0

        print M.sum(axis=0)

        np.set_printoptions(precision=2, suppress=True)
        print M

        vals, vecs = linalg.eig(M)

        v = vecs[:, np.abs(vals - 1).argmin()]
        print v.real / v.real.sum()

####################################################################################################
#
#   Example
#
####################################################################################################


if __name__=='__main__':

    print(KineticsProp.__doc__)

    print KineticsProp(
        # Pulses characterization
        pump_central=625.,
        pump_bw=40.,

        dump_central=835.,
        dump_bw=10.,

        beam_diameter=200.,

        T_max=3.5,
        T_steps=1000,

        A_41_Pr=1 / .150,
        A_23_Pr=1 / .150,
        A_35_Pr=1 / 68.5,
        A_34_Pr=2.5 / 68.5,

        A_41_Pfr=1 / .050,
        A_23_Pfr=1 / .050,
        A_35_Pfr=1 / 2.5,
        A_34_Pfr=2.0,

        Iterations=51,
    )(
        (0.25, 0.0, .100, .150, 0.5, 0.5251)
    )