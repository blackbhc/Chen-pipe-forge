import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic_2d as bin2d
from scipy.interpolate import interp1d

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 26,
        "mathtext.fontset": "cm",
    }
)


class snapshot_utils(object):
    """
    My utils used to analyze a snapshot.
    """

    def __init__(self) -> None:
        pass

    def getStableCenter(self, coordinates, iterMaxTimes=25, encloseRadius=100):
        """
        Get the center of mass as a 1d len=3 array of an coordinates array, asscoiated with optional masses.
        """
        com = np.zeros(3)  # initial value of the center of mass
        for i in range(iterMaxTimes):
            old = 1 * com  # back up the old value
            index = np.where(
                np.linalg.norm(coordinates - com, axis=1, ord=2) < encloseRadius
            )[0]
            com = np.mean(coordinates[index], axis=0)  # get the new value
            err = np.linalg.norm(com - old)  # error distance
            if (
                err < 0.01 * encloseRadius and err < 0.1
            ):  # error < relative value and a absolute value
                break
        return com  # return the value

    def getPrincipleAxes(self, coordinates, masses):
        """
        Get the system's princle axes, which are defined as the eigen vectors of the inertia tensor.
        """
        inertiaTensor = np.zeros(shape=(3, 3))
        rs = np.linalg.norm(coordinates, axis=1, ord=2)
        xs, ys, zs = (
            coordinates[:, 0] / rs,
            coordinates[:, 1] / rs,
            coordinates[:, 2] / rs,
        )
        cross12 = np.sum(masses * xs * ys)
        cross13 = np.sum(masses * xs * zs)
        cross23 = np.sum(masses * ys * zs)
        inertiaTensor[0, 1] = inertiaTensor[1, 0] = cross12
        inertiaTensor[0, 2] = inertiaTensor[2, 0] = cross13
        inertiaTensor[1, 2] = inertiaTensor[2, 1] = cross23
        inertiaTensor[0, 0] = np.sum(masses * (ys**2 + zs**2))
        inertiaTensor[1, 1] = np.sum(masses * (xs**2 + zs**2))
        inertiaTensor[2, 2] = np.sum(masses * (xs**2 + ys**2))
        eigenValues, eigenVectors = np.linalg.eig(inertiaTensor)
        sortIDs = np.argsort(
            np.abs(eigenValues)
        )  # the id to sort the eigenvalues, which mean the inertia moments relative to the eigen vectors
        return eigenVectors.T[sortIDs].T

    def alignDisk(
        self,
        coordinates,
        velocities,
        masses,
        haloCoordinates=[],
        haloVelocities=[],
        encloseRadius=-1,
    ):
        """
        Align the disk plane to the Oxy plane, through aligning the total angular to z axis.
        """
        if encloseRadius <= 0:
            indexes = list(range(0, len(masses), 1))
        else:
            indexes = np.where(
                np.linalg.norm(coordinates, ord=2, axis=1) < encloseRadius
            )[0]
        linearMoment = (
            np.column_stack((masses, masses, masses)) * velocities
        )  # linear moments
        Leach = np.cross(
            coordinates[indexes], linearMoment[indexes], axis=1
        )  # angular moment of each particle
        Ltot = np.sum(Leach, axis=0)  # total angular moment

        newZ = Ltot / np.linalg.norm(Ltot)  # new z axis

        # new y axis: the normal vector for the intersection line between the new Oxy plane and old x=0 plane
        tmp = -newZ[1] / newZ[2]  # the z-comp of the new y axis
        newY = np.array([0, 1, tmp])  # new y axis
        newY = newY / np.linalg.norm(newY)  # normalization

        newX = np.cross(newY, newZ)  # new x axis by cross product
        newX = newX / np.linalg.norm(newX)  # normalization

        rotation = np.column_stack((newX, newY, newZ)).T

        if len(haloCoordinates) == 0:
            coordinates_ = np.matmul(rotation, coordinates.T).T
            velocities_ = np.matmul(rotation, velocities.T).T
            return coordinates_, velocities_
        else:
            # if given halo datasets, also rotate them
            haloCoordinates_ = np.matmul(rotation, haloCoordinates.T).T
            haloVelocities_ = np.matmul(rotation, haloVelocities.T).T
            coordinates_ = np.matmul(rotation, coordinates.T).T
            velocities_ = np.matmul(rotation, velocities.T).T
            return coordinates_, velocities_, haloCoordinates_, haloVelocities_

    def radial_profile_surface_density(
        self,
        coordinates,
        masses,
        Rmin=0.1,
        Rmax=20,
        RbinNum=50,
        PhiBinNum=32,
    ):
        """
        Calculate the radial profile of the azimuthally averaged surface density.
        """
        # calculate the radii and azimuthal angles
        Rs = np.linalg.norm(coordinates[:, :2], axis=1, ord=2)
        Phis = np.arctan2(coordinates[:, 1], coordinates[:, 0])

        # calculate the total masses in different pixels
        massSums = bin2d(
            x=Rs,
            y=Phis,
            values=masses,
            range=[[Rmin, Rmax], [0, np.pi * 2]],
            bins=[RbinNum, PhiBinNum],
            statistic="sum",
        )[0]

        massSums = np.mean(massSums, axis=1)  # azimuthal averages

        rEdges = np.linspace(Rmin, Rmax, RbinNum + 1)  # bin edges of radius
        rs = (rEdges[1:] + rEdges[:-1]) / 2  # central radii of the bins
        deltaR = (Rmax - Rmin) / RbinNum  # bin width of radius
        areas = 2 * np.pi * rs * deltaR / PhiBinNum  # areas of each bin
        surfaceDensity = massSums / areas
        return rs, surfaceDensity

    def radial_profile(
        self,
        coordinates,
        values,
        Rmin=0.1,
        Rmax=20,
        RbinNum=50,
        PhiBinNum=32,
    ):
        """
        Calculate the radial profile of the azimuthal averages for some quantity.
        """
        # calculate the radii and azimuthal angles
        Rs = np.linalg.norm(coordinates[:, :2], axis=1, ord=2)
        Phis = np.arctan2(coordinates[:, 1], coordinates[:, 0])

        # calculate the means in different pixels
        means = bin2d(
            x=Rs,
            y=Phis,
            values=values,
            range=[[Rmin, Rmax], [0, np.pi * 2]],
            bins=[RbinNum, PhiBinNum],
            statistic=np.nanmean,
        )[0]
        means = np.nanmean(means, axis=1)  # azimuthal averages

        rEdges = np.linspace(Rmin, Rmax, RbinNum + 1)  # bin edges of radius
        rs = (rEdges[1:] + rEdges[:-1]) / 2  # central radii of the bins
        return rs, means

    def view_snapshot(
        self,
        coordinates,
        size=20,
        binNum=100,
        ratio=1,
        vmin=None,
        vmax=None,
        tickNum1=9,
        tickNum2=9,
        interpolation="none",
        colorbarLabel="",
        showContour=True,
        contourLevels=9,
        showFig=False,
        saveToDir="",
        t=-1,
        Rbar=-1,
        A2=-1,
    ):
        """
        Plot the image of a snapshot.
        """
        # define the parameters of the figure
        basic = 6
        wCbar = 0.1
        w = basic
        hFaceOn = basic
        hEdgeOn = basic * ratio
        leftMargin = 2
        rightMargin = 1.7
        lowerMargin = 1
        upperMargin = 0.5
        W = w + hEdgeOn + leftMargin + rightMargin
        H = hFaceOn + hEdgeOn + lowerMargin + upperMargin
        fig = plt.figure(figsize=(W, H))
        axFace = fig.add_axes(
            [leftMargin / W, (lowerMargin + hEdgeOn) / H, w / W, hFaceOn / H]
        )
        axLower = fig.add_axes([leftMargin / W, lowerMargin / H, w / W, hEdgeOn / H])
        axRighter = fig.add_axes(
            [
                (leftMargin + w) / W,
                (lowerMargin + hEdgeOn) / H,
                hEdgeOn / W,
                hFaceOn / H,
            ]
        )
        axCbar = fig.add_axes(
            [
                (leftMargin + w + hEdgeOn + 0.25 * rightMargin) / W,
                lowerMargin / H,
                wCbar / W,
                (hFaceOn + hEdgeOn) / H,
            ]
        )

        # functions that transforms the physical value to the pixel values
        def phy2pixel_x(data):
            return (data - -size) / (2 * size) * (binNum - 1)

        def phy2pixel_yz(data):
            return (data - -size * ratio) / (2 * size * ratio) * (binNum * ratio - 1)

        # function to calculate the logarithm of a 2D matrix, which is later used to calculate log(Sigma)
        def logNormMat(matrix):
            mat = 1 * matrix
            index = np.where(mat < 1)
            mat[index] = 1
            mat = np.log10(mat)
            mat[index] = None
            return mat

        # x-y image
        imageXY = bin2d(
            x=coordinates[:, 1],
            y=coordinates[:, 0],
            values=coordinates[:, 0],
            range=[[-size, size], [-size, size]],
            bins=binNum,
            statistic="count",
        )[0]
        imageXY = logNormMat(imageXY)
        axFace.imshow(
            imageXY,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="jet",
            interpolation=interpolation,
        )

        # x-z image
        imageXZ = bin2d(
            x=coordinates[:, 2],
            y=coordinates[:, 0],
            values=coordinates[:, 0],
            range=[[-size * ratio, size * ratio], [-size, size]],
            bins=[int(binNum * ratio), binNum],
            statistic="count",
        )[0]
        imageXZ = logNormMat(imageXZ)
        axLower.imshow(
            imageXZ,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="jet",
            interpolation=interpolation,
        )

        # y-z image
        imageYZ = bin2d(
            x=coordinates[:, 1],
            y=coordinates[:, 2],
            values=coordinates[:, 0],
            range=[[-size, size], [-size * ratio, size * ratio]],
            bins=[binNum, int(binNum * ratio)],
            statistic="count",
        )[0]
        imageYZ = logNormMat(imageYZ)
        im = axRighter.imshow(
            imageYZ,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="jet",
            interpolation=interpolation,
        )

        # plot the iso-density contours
        if showContour:
            contourX, contourY = np.meshgrid(
                np.arange(0, imageXY.shape[0]), np.arange(0, imageXY.shape[1])
            )
            axFace.contour(
                contourX,
                contourY,
                imageXY,
                levels=contourLevels,
                colors="black",
                alpha=0.5,
                linewidths=1,
                origin="lower",
            )
            contourZ, contourY = np.meshgrid(
                np.arange(0, imageYZ.shape[1]), np.arange(0, imageYZ.shape[0])
            )
            axRighter.contour(
                contourZ,
                contourY,
                imageYZ,
                levels=contourLevels,
                colors="black",
                alpha=0.5,
                linewidths=1,
                origin="lower",
            )
            contourX, contourZ = np.meshgrid(
                np.arange(0, imageXZ.shape[1]), np.arange(0, imageXZ.shape[0])
            )
            axLower.contour(
                contourX,
                contourZ,
                imageXZ,
                levels=contourLevels,
                colors="black",
                alpha=0.5,
                linewidths=1,
                origin="lower",
            )

        # calculate the ticks
        ticks1 = np.around(np.linspace(-size, size, tickNum1), 1)
        ticks2 = np.around(np.linspace(-size * ratio, size * ratio, tickNum2), 1)
        # setup the ticks
        axFace.set_xticks(phy2pixel_x(ticks1), [])
        axFace.set_yticks(phy2pixel_x(ticks1[1:]), ticks1[1:])
        axLower.set_xticks(phy2pixel_x(ticks1), ticks1)
        axLower.set_yticks(phy2pixel_yz(ticks2), ticks2)
        axRighter.set_xticks(phy2pixel_yz(ticks2[1:]), ticks2[1:])
        axRighter.set_yticks(phy2pixel_x(ticks1), [])
        # set up the labels of axes
        axFace.set_ylabel(r"$Y$ [kpc]")
        axFace.set_xlabel(r"$X$ [kpc]")
        axLower.set_xlabel(r"$X$ [kpc]")
        axLower.set_ylabel(r"$Z$ [kpc]")
        axRighter.set_xlabel(r"$Z$ [kpc]")

        # show the time of the snapshot
        showText = ""
        if t >= 0:
            showText += f"t={t:.2f} Gyr"
        if A2 >= 0:
            showText += f"\nA2={A2:.2f}"
        if len(showText) > 0:
            axLower.text(
                phy2pixel_x(1.5 * size), phy2pixel_yz(0), showText, ma="center"
            )
        # plot the colorbar
        plt.colorbar(mappable=im, cax=axCbar, label=colorbarLabel)

        # plot a circle for Rbar
        if Rbar > 0:
            thetas = np.linspace(0, np.pi * 2, 72)
            plotXs = Rbar * np.cos(thetas)
            plotYs = Rbar * np.sin(thetas)
            axFace.plot(phy2pixel_x(plotXs), phy2pixel_x(plotYs), "r-")

        # save or show the figure if necessary
        if saveToDir != "":
            plt.savefig(saveToDir)
        if showFig:
            plt.show()
        plt.close(fig)

    def view_with_values(
        self,
        coordinates,
        values,
        statistic,
        size=20,
        binNum=100,
        ratio=1,
        vmin=None,
        vmax=None,
        tickNum1=9,
        tickNum2=9,
        interpolation="none",
        colorbarLabel="",
        showContour=True,
        contourLevels=9,
        showFig=False,
        saveToDir="",
        t=-1,
        Rbar=-1,
        A2=-1,
    ):
        """
        Plot the image of a snapshot, color coded by some value.
        """
        # define the parameters of the figure
        basic = 6
        wCbar = 0.1
        w = basic
        hFaceOn = basic
        hEdgeOn = basic * ratio
        leftMargin = 1.5
        rightMargin = 1.7
        lowerMargin = 1
        upperMargin = 0.5
        W = w + hEdgeOn + leftMargin + rightMargin
        H = hFaceOn + hEdgeOn + lowerMargin + upperMargin
        fig = plt.figure(figsize=(W, H))
        axFace = fig.add_axes(
            [leftMargin / W, (lowerMargin + hEdgeOn) / H, w / W, hFaceOn / H]
        )
        axLower = fig.add_axes([leftMargin / W, lowerMargin / H, w / W, hEdgeOn / H])
        axRighter = fig.add_axes(
            [
                (leftMargin + w) / W,
                (lowerMargin + hEdgeOn) / H,
                hEdgeOn / W,
                hFaceOn / H,
            ]
        )
        axCbar = fig.add_axes(
            [
                (leftMargin + w + hEdgeOn + 0.25 * rightMargin) / W,
                lowerMargin / H,
                wCbar / W,
                (hFaceOn + hEdgeOn) / H,
            ]
        )

        # functions that transforms the physical value to the pixel values
        def phy2pixel_x(data):
            return (data - -size) / (2 * size) * (binNum - 1)

        def phy2pixel_yz(data):
            return (data - -size * ratio) / (2 * size * ratio) * (binNum * ratio - 1)

        # x-y image
        imageXY = bin2d(
            x=coordinates[:, 1],
            y=coordinates[:, 0],
            values=values,
            range=[[-size, size], [-size, size]],
            bins=binNum,
            statistic=statistic,
        )[0]
        axFace.imshow(
            imageXY,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="jet",
            interpolation=interpolation,
        )

        # x-z image
        imageXZ = bin2d(
            x=coordinates[:, 2],
            y=coordinates[:, 0],
            values=values,
            range=[[-size * ratio, size * ratio], [-size, size]],
            bins=[int(binNum * ratio), binNum],
            statistic=statistic,
        )[0]
        axLower.imshow(
            imageXZ,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="jet",
            interpolation=interpolation,
        )

        # y-z image
        imageYZ = bin2d(
            x=coordinates[:, 1],
            y=coordinates[:, 2],
            values=values,
            range=[[-size, size], [-size * ratio, size * ratio]],
            bins=[binNum, int(binNum * ratio)],
            statistic=statistic,
        )[0]
        im = axRighter.imshow(
            imageYZ,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="jet",
            interpolation=interpolation,
        )

        # plot the iso-density contours
        if showContour:
            # function to calculate the logarithm of a 2D matrix, which is later used to calculate log(Sigma)
            def logNormMat(matrix):
                mat = 1 * matrix
                index = np.where(mat < 1)
                mat[index] = 1
                mat = np.log10(mat)
                mat[index] = None
                return mat

            # density X-Y
            imageXY = bin2d(
                x=coordinates[:, 1],
                y=coordinates[:, 0],
                values=coordinates[:, 0],
                range=[[-size, size], [-size, size]],
                bins=binNum,
                statistic="count",
            )[0]
            imageXY = logNormMat(imageXY)
            contourX, contourY = np.meshgrid(
                np.arange(0, imageXY.shape[0]), np.arange(0, imageXY.shape[1])
            )
            axFace.contour(
                contourX,
                contourY,
                imageXY,
                levels=contourLevels,
                colors="black",
                alpha=0.5,
                linewidths=1,
                origin="lower",
            )
            # X-Z image
            imageXZ = bin2d(
                x=coordinates[:, 2],
                y=coordinates[:, 0],
                values=coordinates[:, 0],
                range=[[-size * ratio, size * ratio], [-size, size]],
                bins=[int(binNum * ratio), binNum],
                statistic="count",
            )[0]
            imageXZ = logNormMat(imageXZ)
            contourZ, contourY = np.meshgrid(
                np.arange(0, imageYZ.shape[1]), np.arange(0, imageYZ.shape[0])
            )
            axRighter.contour(
                contourZ,
                contourY,
                imageYZ,
                levels=contourLevels,
                colors="black",
                alpha=0.5,
                linewidths=1,
                origin="lower",
            )
            # Y-Z image
            imageYZ = bin2d(
                x=coordinates[:, 1],
                y=coordinates[:, 2],
                values=coordinates[:, 0],
                range=[[-size, size], [-size * ratio, size * ratio]],
                bins=[binNum, int(binNum * ratio)],
                statistic="count",
            )[0]
            imageYZ = logNormMat(imageYZ)
            contourX, contourZ = np.meshgrid(
                np.arange(0, imageXZ.shape[1]), np.arange(0, imageXZ.shape[0])
            )
            axLower.contour(
                contourX,
                contourZ,
                imageXZ,
                levels=contourLevels,
                colors="black",
                alpha=0.5,
                linewidths=1,
                origin="lower",
            )

        # calculate the ticks
        ticks1 = np.around(np.linspace(-size, size, tickNum1), 1)
        ticks2 = np.around(np.linspace(-size * ratio, size * ratio, tickNum2), 1)
        # setup the ticks
        axFace.set_xticks(phy2pixel_x(ticks1), [])
        axFace.set_yticks(phy2pixel_x(ticks1[1:]), ticks1[1:])
        axLower.set_xticks(phy2pixel_x(ticks1), ticks1)
        axLower.set_yticks(phy2pixel_yz(ticks2), ticks2)
        axRighter.set_xticks(phy2pixel_yz(ticks2[1:]), ticks2[1:])
        axRighter.set_yticks(phy2pixel_x(ticks1), [])
        # set up the labels of axes
        axFace.set_ylabel(r"$Y$ [kpc]")
        axFace.set_xlabel(r"$X$ [kpc]")
        axLower.set_xlabel(r"$X$ [kpc]")
        axLower.set_ylabel(r"$Z$ [kpc]")
        axRighter.set_xlabel(r"$Z$ [kpc]")

        # show the time of the snapshot
        showText = ""
        if t >= 0:
            showText += f"t={t:.2f} Gyr"
        if A2 >= 0:
            showText += f"\nA2={A2:.2f}"
        if len(showText) > 0:
            axLower.text(
                phy2pixel_x(1.5 * size), phy2pixel_yz(0), showText, ma="center"
            )
        # plot the colorbar
        plt.colorbar(mappable=im, cax=axCbar, label=colorbarLabel)

        # plot a circle for Rbar
        if Rbar > 0:
            thetas = np.linspace(0, np.pi * 2, 72)
            plotXs = Rbar * np.cos(thetas)
            plotYs = Rbar * np.sin(thetas)
            axFace.plot(phy2pixel_x(plotXs), phy2pixel_x(plotYs), "r-")

        # save or show the figure if necessary
        if saveToDir != "":
            plt.savefig(saveToDir)
        if showFig:
            plt.show()
        plt.close(fig)

    def A2(self, phis, masses=[], normalize=True):
        """
        Calculate the amplitude of the m=2 mode.
        """
        exponents = np.exp(2j * phis)

        if len(masses) == 0:
            A0 = len(phis)
            A2 = np.sum(exponents)
        else:
            A0 = np.sum(masses)
            A2 = np.sum(masses * exponents)

        if normalize:
            return np.abs(A2 / A0)
        else:
            return np.abs(A2)

    def Sbuckling(self, phis, Zs, masses=[], normalize=True):
        """
        Calculate the amplitude of the m=2 mode.
        """
        exponents = np.exp(2j * phis)

        if len(masses) == 0:
            A0 = len(phis)
            numerator = np.sum(Zs * exponents)
        else:
            A0 = np.sum(masses)
            numerator2 = np.sum(masses * Zs * exponents)

        if normalize:
            return np.abs(numerator / A0)
        else:
            return np.abs(numerator)

    def m2phase(self, phis, masses=[]):
        """
        Calculate the phase of the m=2 mode.
        """
        exponents = np.exp(2j * phis)

        if len(masses) == 0:
            A2 = np.sum(exponents)
        else:
            A2 = np.sum(masses * exponents)

        return np.angle(A2) / 2

    def A2profile(
        self, phis, rs, masses=[], Rmin=0.1, Rmax=20, RbinNum=40, normalize=True
    ):
        """
        Calculate the radial profile of the m=2 amplitudes.
        """
        RbinEdges = np.linspace(Rmin, Rmax, RbinNum + 1)
        A2s = []
        for i in range(RbinNum):
            index = np.where((rs >= RbinEdges[i]) & (rs < RbinEdges[i + 1]))[0]
            A2s.append(self.A2(phis=phis[index], masses=masses, normalize=normalize))
        return np.array(A2s)

    def RbarThreshold(
        self, phis, rs, masses=[], Rmin=0.01, Rmax=20, RbinNum=40, threshold=1
    ):
        """
        Calculate the bar radius that A2 reach some threshold, 1 for max.
        """
        # calculate the A2 profile
        profile = self.A2profile(
            phis=phis, rs=rs, masses=masses, Rmin=Rmin, Rmax=Rmax, RbinNum=RbinNum
        )
        Rlocs = np.linspace(Rmin, Rmax, RbinNum + 1)
        Rlocs = (Rlocs[1:] + Rlocs[:-1]) / 2
        interpolateKernel = interp1d(Rlocs, profile, kind="linear")
        finerRs = np.linspace(
            Rlocs.min(), Rlocs.max(), 5 * RbinNum
        )  # finer radial bins
        inerpolatedProfile = interpolateKernel(finerRs)  # smoothed A2 at the finer bins
        maxID = np.argmax(inerpolatedProfile)  # location of the maximal A2
        # critical values of A2
        Max = inerpolatedProfile[maxID]
        outerMin = np.min(inerpolatedProfile[maxID:])
        outerRange = Max - outerMin  # range
        thresholdA2 = outerMin + outerRange * threshold  # the effective threshold
        locID = np.where(inerpolatedProfile[maxID:] <= thresholdA2)[0][0]
        return finerRs[maxID:][locID]

    def inner_product(self, v1, v2):
        return np.sum(v1 * v2, axis=1)

    def car2sph(self, coordinates, velocities):
        """
        Transform the coordinates and velocities from the cartesian coordinates to spherical coordinates.
        Return: the transformed coordinates and velocities, all in the (r, phi, theta) order.
        """
        rs = np.linalg.norm(coordinates, axis=1)  # spherical coordinates
        unit_r = coordinates / np.column_stack((rs, rs, rs))  # unit radial vector
        unit_z = np.column_stack(
            (np.zeros(len(rs)), np.zeros(len(rs)), np.ones(len(rs)))
        )
        phi = np.atan2(coordinates[:, 1], coordinates[:, 0])  # azimuthal angles
        theta = np.arccos(self.inner_product(unit_z, unit_r))
        unit_phi = np.column_stack(
            (-np.sin(phi), np.cos(phi), np.zeros(len(phi)))
        )  # unit azimuthal vector
        Vr = self.inner_product(unit_r, velocities)  # radial velocity
        Vphi = self.inner_product(unit_phi, velocities)  # azimuthal velocity
        vecVr = np.column_stack((Vr, Vr, Vr)) * unit_r
        vecVphi = np.column_stack((Vphi, Vphi, Vphi)) * unit_phi
        Vtheta = velocities - vecVr - vecVphi
        oCoordinates = np.column_stack((rs, phi, theta))
        oVeloicties = np.column_stack((Vr, Vphi, Vtheta))
        return oCoordinates, oVeloicties

    def car2cyl(self, coordinates, velocities):
        """
        Transform the coordinates and velocities from the cartesian coordinates to spheical coordinates.
        Return: the transformed coordinates and velocities, all in the (R, phi, z) order.
        """
        Rs = np.linalg.norm(coordinates[:, :2], axis=1)  # spherical coordinates
        unit_R = coordinates / np.column_stack((Rs, Rs, Rs))  # unit radial vector
        phi = np.atan2(coordinates[:, 1], coordinates[:, 0])  # azimuthal angles
        unit_phi = np.column_stack(
            (-np.sin(phi), np.cos(phi), np.zeros(len(phi)))
        )  # unit azimuthal vector
        VR = self.inner_product(unit_R, velocities)  # radial velocity
        Vphi = self.inner_product(unit_phi, velocities)  # azimuthal velocity
        vecVr = np.column_stack((VR, VR, VR)) * unit_R
        vecVphi = np.column_stack((Vphi, Vphi, Vphi)) * unit_phi
        oCoordinates = np.column_stack((Rs, phi, coordinates[:, 2]))
        oVeloicties = np.column_stack((VR, Vphi, velocities[:, 2]))
        return oCoordinates, oVeloicties
