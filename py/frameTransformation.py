import numpy as np
from sklearn.preprocessing import normalize


class frameTransformer:
    """
    The coordinate transformer.
    """

    def car2cyl(self, posvels):
        """
        Transform the 6D vector from cartesian coordinates into cylindrical coordinates.
        The input should be N*6 array.
        Return: the transformed coordinates and velocities.
        """
        coordinates = posvels[:, :3]
        velocities = posvels[:, 3:]
        complex_planar_pos = coordinates[:, 0] + 1j * coordinates[:, 1]
        Rs = np.abs(complex_planar_pos)
        Phis = np.angle(complex_planar_pos)
        Zs = coordinates[:, 2]

        index = np.where(Rs < 1e-32)[0]
        Rs[index] = 1  # avoid 0 division
        VRs = (
            coordinates[:, 0] * velocities[:, 0] + coordinates[:, 1] * velocities[:, 1]
        ) / Rs
        VPhis = (
            coordinates[:, 0] * velocities[:, 1] - coordinates[:, 1] * velocities[:, 0]
        ) / Rs
        Rs[index] = VRs[index] = VPhis[index] = 0
        Vzs = velocities[:, 2]
        return np.column_stack((Rs, Phis, Zs, VRs, VPhis, Vzs))

    def cyl2car(self, posvels):
        """
        Transform the 6D vector from cylindrical coordinates into cartesian coordinates.
        The input should be N*6 array.
        Return: the transformed coordinates and velocities.
        """
        Rs = posvels[:, 0]
        Phis = posvels[:, 1]
        Zs = posvels[:, 2]
        VRs = posvels[:, 3]
        VPhis = posvels[:, 4]
        VZs = posvels[:, 5]
        Xs = np.cos(Phis) * Rs
        Ys = np.sin(Phis) * Rs
        VXs = VRs * np.cos(Phis) + VPhis * np.cos(Phis + np.pi / 2)
        VYs = VRs * np.sin(Phis) + VPhis * np.sin(Phis + np.pi / 2)
        return np.column_stack((Xs, Ys, Zs, VXs, VYs, VZs))

    def car2sph(self, posvels):
        """
        Transform the 6D vector from cartesian coordinates into spherical coordinates.
        The input should be N*6 array.
        Return: the transformed coordinates and velocities.
        """
        coordinates = posvels[:, :3]
        velocities = posvels[:, 3:]

        # phi for longtitude and theta for latitude
        unitRs = normalize(coordinates, axis=1)
        Rs = np.linalg.norm(coordinates, ord=2, axis=1)
        Phis = np.atan2(coordinates[:, 1], coordinates[:, 0])
        unitPhis = normalize(
            np.column_stack(
                (-coordinates[:, 1], coordinates[:, 0], np.zeros(len(coordinates)))
            )
        )
        unitThetas = np.cross(unitPhis, unitRs, axis=1)

        index = np.where(Rs < 1e-32)[0]
        Rs[index] = 1  # avoid 0 division
        Thetas = np.acos(coordinates[:, 2] / Rs)
        VRs = np.sum(velocities * unitRs, axis=1)
        # longitudinal velocity
        VPhis = np.sum(velocities * unitPhis, axis=1)
        # latitudinal velocity
        VThetas = np.sum(velocities * unitThetas, axis=1)
        Rs[index] = Thetas[index] = 0
        return np.column_stack((Rs, Thetas, Phis, VRs, VThetas, VPhis))

    def sph2car(self, posvels):
        """
        Transform the 6D vector from spherical coordinates into cartesian coordinates.
        The input should be N*6 array.
        Return: the transformed coordinates and velocities.
        """
        Rs = posvels[:, 0]
        Thetas = posvels[:, 1]
        Phis = posvels[:, 2]
        VRs = posvels[:, 3]
        VThetas = posvels[:, 4]
        VPhis = posvels[:, 5]
        Xs = np.cos(Phis) * Rs * np.sin(Thetas)
        Ys = np.sin(Phis) * Rs * np.sin(Thetas)
        Zs = np.cos(Thetas) * Rs

        VXs = (
            VRs * np.sin(Thetas) * np.cos(Phis)
            + VPhis * np.cos(Phis + np.pi / 2)
            + VThetas * np.sin(Thetas + np.pi / 2) * np.cos(Phis)
        )
        VYs = (
            VRs * np.sin(Thetas) * np.sin(Phis)
            + VPhis * np.sin(Phis + np.pi / 2)
            + VThetas * np.sin(Thetas + np.pi / 2) * np.sin(Phis)
        )
        VZs = VRs * np.cos(Thetas) + VThetas * np.cos(Thetas + np.pi / 2)
        return np.column_stack((Xs, Ys, Zs, VXs, VYs, VZs))


class anisotropy(frameTransformer):
    """
    Calculator of the anisotropy parameter, a demo:

    test_calculator = anisotropy(Type="dispersion")

    ts = np.linspace(0, np.pi * 4, 1001)
    omega = 1
    phis = omega * ts
    rawPos = np.column_stack((np.cos(phis), np.sin(phis), 0 * ts))
    pos = (rawPos[1:] + rawPos[:-1]) / 2
    vel = (rawPos[1:] - rawPos[:-1]) / np.column_stack(
        ((ts[1:] - ts[:-1]), (ts[1:] - ts[:-1]), (ts[1:] - ts[:-1]))
    )
    pure_circular_posvels = np.column_stack((pos, vel))

    test_calculator.cal(pure_circular_posvels)
    """

    def __init__(self, Type):
        if type(Type) is not str:
            raise ("The type should be str, for 'dispersion' or 'velocity'!")
        self.Type = Type.lower()
        if self.Type != "dispersion" and self.Type != "velocity":
            raise ("The type should be str, for 'dispersion' or 'velocity'!")

    def cal(self, posvels):
        """
        Calculate the anisotropy of the given data sets.
        """
        spherical_posvels = self.car2sph(posvels)
        VRs = spherical_posvels[:, 3]
        VPhis = spherical_posvels[:, 4]
        VThetas = spherical_posvels[:, 5]
        denominator = 0
        numerator = 0
        if self.Type == "velocity":
            numerator = np.mean(VPhis) ** 2 + np.mean(VThetas) ** 2
            denominator = 2 * np.mean(VRs) ** 2
        else:  # dispersion case
            numerator = np.std(VPhis) ** 2 + np.std(VThetas) ** 2
            denominator = 2 * np.std(VRs) ** 2

        if denominator < 1e-16:
            return -np.inf
        else:
            return 1 - numerator / denominator
