import math
import numpy as np
import matplotlib.pyplot as plt


class Droplet:
    def __init__(self):
        self.radius = None
        self.diameter = None
        self.mass_daltons = None
        self.volume_in_liters = None
        self.volume_in_milliliters = None
        self.rayleigh_charge = None
        self.analyte_concentration_millimolar = None
        self.analyte_molecule_count = None
        self.density = None
        self.water_molecules = None

    def print_params(self):
        print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Tip produces initial droplets of diemeter " + str(self.diameter) + " nm")
        print("Droplet volume " + str(self.volume_in_liters) + " L (" + str(self.volume_in_milliliters) + " mL)")
        print("Expected mass of initial droplet is " + str(self.mass_daltons / 1000000) + " MDa")
        print("Expected charge of initial droplet is " + str(self.rayleigh_charge) + " e")
        print("Analyte molecules per initial droplet: " + str(self.analyte_molecule_count))
        print("Water molecules in droplet: " + str(self.water_molecules) + " molecules")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++\n")


def initial_molecules_per_drop_tip_size_given(tip_diameter_nm, analyte_concentration_mM, size_scaler=1 / 17, density=1):
    # Calculates the analyte molecules per droplet when the droplet size is given in nanometers
    # Analyte concentration in mM
    avo_number = 6.023e23
    g_to_Da = 1.6605e-24
    surfacet = 0.07286
    coul_e = 1.6022E-19
    permittivity = 8.8542E-12  # vacuum permittivity
    mass_h2o = 18.01528

    droplet = Droplet()
    droplet.analyte_concentration_millimolar = analyte_concentration_mM
    droplet.diameter = tip_diameter_nm * size_scaler
    droplet.radius = droplet.diameter / 2
    droplet.density = density
    droplet.rayleigh_charge = (8 * np.pi * np.sqrt(permittivity * surfacet * np.power((droplet.diameter * 1.0E-9) / 2,
                                                                                      3))) / coul_e
    droplet.volume_in_liters = ((4 / 3) * np.pi * np.power(droplet.diameter / 2000000000, 3)) * 1000
    droplet.volume_in_milliliters = droplet.volume_in_liters * 1000
    droplet.mass_daltons = (droplet.volume_in_milliliters * droplet.density) / g_to_Da
    droplet.analyte_molecule_count = (droplet.analyte_concentration_millimolar / 1000) * droplet.volume_in_liters * avo_number
    droplet.water_molecules = droplet.mass_daltons / mass_h2o

    return droplet


def rayleigh_calculator_diameter_given(diameter_nm, density=1):
    # Calculates the analyte molecules per droplet when the droplet size is given in nanometers
    # Analyte concentration in mM
    avo_number = 6.023e23
    g_to_Da = 1.6605e-24
    surfacet = 0.07286
    coul_e = 1.6022E-19
    permittivity = 8.8542E-12  # vacuum permittivity
    mass_h2o = 18.01528

    droplet = Droplet()
    droplet.diameter = diameter_nm
    droplet.radius = droplet.diameter / 2
    droplet.density = density
    droplet.rayleigh_charge = (8 * np.pi * np.sqrt(permittivity * surfacet * np.power((droplet.diameter * 1.0E-9) / 2,
                                                                                      3))) / coul_e
    droplet.volume_in_liters = ((4 / 3) * np.pi * np.power(droplet.diameter / 2000000000, 3)) * 1000
    droplet.volume_in_milliliters = droplet.volume_in_liters * 1000
    droplet.mass_daltons = (droplet.volume_in_milliliters * droplet.density) / g_to_Da
    droplet.water_molecules = droplet.mass_daltons / mass_h2o

    return droplet

def rayleigh_calculator_charge_given(charge, density=1):
    # Calculates the analyte molecules per droplet when the droplet size is given in nanometers
    # Analyte concentration in mM
    avo_number = 6.023e23
    g_to_Da = 1.6605e-24
    surfacet = 0.07286
    coul_e = 1.6022E-19
    permittivity = 8.8542E-12  # vacuum permittivity
    mass_h2o = 18.01528

    droplet = Droplet()
    droplet.rayleigh_charge = charge
    droplet.density = density
    droplet.diameter = (np.power(((np.power(((droplet.rayleigh_charge * coul_e) / (8 * np.pi)), 2)) / (permittivity *
                                                                                             surfacet)), 1/3) * (2 / 1.0e-9))
    droplet.radius = droplet.diameter / 2

    droplet.volume_in_liters = ((4 / 3) * np.pi * np.power(droplet.diameter / 2000000000, 3)) * 1000
    droplet.volume_in_milliliters = droplet.volume_in_liters * 1000
    droplet.mass_daltons = (droplet.volume_in_milliliters * droplet.density) / g_to_Da
    droplet.water_molecules = droplet.mass_daltons / mass_h2o

    return droplet


if __name__ == "__main__":
    droplet = initial_molecules_per_drop_tip_size_given(700, 5)
    droplet.print_params()
    droplet = initial_molecules_per_drop_tip_size_given(1720, 5)
    droplet.print_params()
    droplet = rayleigh_calculator_diameter_given(26.72)
    droplet.print_params()
    droplet = rayleigh_calculator_charge_given(1)
    droplet.print_params()
    droplet = rayleigh_calculator_charge_given(2)
    droplet.print_params()
    droplet = rayleigh_calculator_charge_given(3)
    droplet.print_params()
