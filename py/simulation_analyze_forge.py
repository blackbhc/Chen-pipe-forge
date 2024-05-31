import numpy as np
import h5py
import warnings


# BUG: currently, onely support models with disk and DM halo
class galactic_snapshot:
    def __init__(self, filename, attr_dict):
        """
        filename: string for the path of the snapshot file.
        attr_dict: dictionary for the attributes of the snapshot file, see below.

        key-value pairs of the attr_dict:
        "type": string for simulation type, only support gadget4 at present.
        "disk": list or list-like, particle type ids for the disk(s) particles.
        "halo": as "disk" but for DM halo(s).
        "bulge": as "disk" but for bugle.
        "stellar_halo": as "disk" but for stellar halo(s).
        """
        self.filename = filename
        self.attr_dict = attr_dict
        self.cal_centered = False
        self.has_info = False
        # TODO: try to remove the following direct calls
        self.disk = {}
        self.halo = {}
        self.disk["ids"] = None
        self.halo["ids"] = None
        self.comps = (self.disk, self.halo)
        # set up the components
        for target in self.attr_dict:
            self.__setup_ids(target)
        # read in the 7D infos (position, velocity, mass) of each components
        self.__read_info()

    def __to_uint_tuple(self, list_like):
        return tuple(np.array(list_like, dtype=np.uint))

    def __setup_ids(self, target):
        # only set up the sets when necessary
        list_like = self.attr_dict[target]
        tuple_like = self.__to_uint_tuple(list_like)
        exec(f"self.{target}['ids'] = tuple_like")

    def __read_info(self):
        file = h5py.File(self.filename, "r")
        for comp in self.comps:
            masses = []
            coordinates = []
            velocities = []
            for id in comp["ids"]:
                coordinates.append(file[f"PartType{id}"]["Coordinates"][...])
                velocities.append(file[f"PartType{id}"]["Velocities"][...])
                masses.append(file[f"PartType{id}"]["Masses"][...])
            comp["masses"] = np.hstack(masses)
            comp["coordinates"] = np.vstack(coordinates)
            comp["velocities"] = np.vstack(velocities)
        file.close()
        self.has_info = True

    def __ensure_set_disk(self):
        # ensure the disk components have been specified
        if not ("disk" in self.attr_dict.keys()):
            raise ("There is no any component has been set as the disk component(s)!")

    def __ensure_has_info(self):
        # ensure the simulation data has been loaded
        if not self.has_info:
            self.__read_info()

    def __update_center(self):
        com = self.get_center()
        self.center = com
        self.cal_centered = True

    def __ensure_caled_center(self):
        if not self.cal_centered:
            self.__update_center()

    def __ensure_everything_ok(self):
        # the all in one wrapper of the ensure functions
        self.__ensure_set_disk()
        self.__ensure_has_info()
        self.__ensure_caled_center()

    def get_center(self, R_enclose=10, tolerance=0.01, max_iter=25):
        """
        Get the galactic center based on the spatial distribution of the disk components.

        Parameters:
        R_enclose: the enclose radius.
        tolerance: threshold for convergence.
        max_iter: int, maximal iteration times.

        Return:
        com: 1x3 np.array.
        """
        # check whether there is disk components
        self.__ensure_set_disk()
        # ensure the data has been loaded
        self.__ensure_has_info()
        # ensure max_iter>0
        if max_iter < 1:
            raise ("Require a positive integer for maximum iteration times.")

        # exclude the possible inf particles
        index = np.where(
            np.linalg.norm(self.disk["coordinates"], axis=1, ord=2) < R_enclose * 100
        )[0]
        com = np.nanmean(self.disk["coordinates"][index], axis=0)

        for i in range(max_iter):
            old_one = com * 1.0
            index = np.where(
                np.linalg.norm(self.disk["coordinates"] - old_one, axis=1, ord=2) < 10
            )[0]
            com = np.nanmean(self.disk["coordinates"][index], axis=0)
            if np.linalg.norm(com - old_one) <= tolerance:
                break

        if i == max_iter:
            warnings.warn("The iteration may not reach a convergence.")

        return com

    def get_A2(self, R_enclose=10):
        """
        Calculate the A2 parameter of the disk components inside some enclosed radius

        Parameters:
        R_enclose: the enclose radius.

        Return:
        A2: complex
        """
        self.__ensure_everything_ok()
        # extract the particles
        masses = self.disk["masses"]
        coordinates = self.disk["coordinates"] - self.center

        index = np.where(
            np.linalg.norm(coordinates - self.center, axis=1, ord=2) < R_enclose
        )[0]

        denominator = np.sum(masses[index])
        phis = np.arctan2(coordinates[:, 1], coordinates[:, 0])
        numerator = np.sum(masses * np.exp(2j * phis))
        return numerator / denominator

    def cal_Jacobi_energy():
        pass
