import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic_2d as bin2d
from scipy.interpolate import interp1d
from scipy.stats import linregress

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
            rs = np.linalg.norm(coordinates - com, axis=1, ord=2)
            com = np.mean(coordinates[rs < encloseRadius], axis=0)  # get the new value
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
        cmap="jet",
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
            cmap=cmap,
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
            cmap=cmap,
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
            cmap=cmap,
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

    def face_on_star_gas(
        self,
        coordinates,
        cmaps=["bone", "bone", "copper"],
        interpolation="none",
        size=20,
        binNum=100,
        vmin=None,
        vmax=None,
        tickNum_x=9,
        tickNum_y=7,
        showContours=[True, False, False],
        contourLevels=9,
        showFig=False,
        saveToDir="",
        t=-1,
        Rbar=-1,
    ):
        """
        Plot the image of a snapshot: face-on views of the old stars, new born stars, and gas.
        -----
        coordinates: list or similiar structure for the old stars, new born stars, and gas.
        cmaps: the corresponding colormap for the old stars, new born stars, and gas.
        """

        # functions that transforms the physical value to the pixel values
        def phy2pixel(data):
            return (data - -size) / (2 * size) * (binNum - 1)

        # function to calculate the logarithm of a 2D matrix, which is later used to calculate log(Sigma)
        def logNormMat(matrix):
            mat = 1 * matrix
            index = np.where(mat < 1)
            mat[index] = 1
            mat = np.log10(mat)
            mat[index] = None
            return mat

        # define the parameters of the figure
        basic = 6
        wCbar = 0.1
        w = basic
        h = basic
        hgap = h * 0.18
        h_cbar = basic * 0.05
        leftMargin = 1.6
        rightMargin = 0.6
        lowerMargin = 1
        upperMargin = 0.4
        W = w * 3 + leftMargin + rightMargin
        H = h + hgap + lowerMargin + upperMargin
        figInit = plt.figure()
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(W, H))

        colorbar_labels = [
            r"$\lg N_{\rm old}$",
            r"$\lg N_{\rm new}$",
            r"$\lg N_{\rm gas}$",
        ]
        for i in range(3):
            ax = axes[1, i]
            cax = axes[0, i]
            ax.set_position(
                [
                    (leftMargin + i * w) / W,
                    lowerMargin / H,
                    w / W,
                    h / H,
                ]
            )
            cax.set_position(
                [
                    (leftMargin + i * w) / W,
                    (lowerMargin + h + hgap) / H,
                    w / W,
                    h_cbar / H,
                ]
            )

            coord = coordinates[i]
            cmap = cmaps[i]
            label = colorbar_labels[i]
            showContour = showContours[i]
            if len(coord) == 0:
                ax.set_xticks([])
                ax.set_yticks([])
                cax.set_xticks([])
                cax.set_yticks([])
                continue  # boundary case for which there is no effective data
            # x-y image
            image = bin2d(
                x=coord[:, 1],
                y=coord[:, 0],
                values=coord[:, 0],
                range=[[-size, size], [-size, size]],
                bins=binNum,
                statistic="count",
            )[0]
            image = logNormMat(image)
            im = ax.imshow(
                image,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                interpolation=interpolation,
            )
            # plot the colorbar
            plt.colorbar(mappable=im, cax=cax, label=label, orientation="horizontal")

            # plot the iso-density contours
            if showContour:
                contourX, contourY = np.meshgrid(
                    np.arange(0, image.shape[0]), np.arange(0, image.shape[1])
                )
                ax.contour(
                    contourX,
                    contourY,
                    image,
                    levels=contourLevels,
                    colors="black",
                    alpha=0.5,
                    linewidths=1,
                    origin="lower",
                )

            # setup the x ticks
            ticks = np.around(np.linspace(-size, size, tickNum_x), 1)
            ax.set_xticks(phy2pixel(ticks[:-1]), ticks[:-1])
            # remove the y ticks
            ax.set_yticks([])
            # set up the labels of axes
            ax.set_xlabel(r"$X$ [kpc]")

            # show the time of the snapshot
            showText = ""
            if t >= 0:
                showText += f"t={t:.2f} Gyr"
                ax.text(
                    phy2pixel(-0.9 * size),
                    phy2pixel(0.85 * size),
                    showText,
                    ma="center",
                    color="red",
                )

            # plot a circle for Rbar
            if Rbar > 0:
                thetas = np.linspace(0, np.pi * 2, 72)
                plotXs = Rbar * np.cos(thetas)
                plotYs = Rbar * np.sin(thetas)
                ax.plot(phy2pixel(plotXs), phy2pixel(plotYs), "r-")

        # reset the last x ticks
        axes[1, 2].set_xticks(phy2pixel(ticks), ticks)
        # set the y ticks and label
        ticks = np.around(np.linspace(-size, size, tickNum_y), 1)
        axes[1, 0].set_yticks(phy2pixel(ticks), ticks)
        axes[1, 0].set_ylabel(r"$Y$ [kpc]")

        # plot a circle for Rbar
        if Rbar > 0:
            thetas = np.linspace(0, np.pi * 2, 72)
            plotXs = Rbar * np.cos(thetas)
            plotYs = Rbar * np.sin(thetas)
            for i in range(3):
                axes[1, i].plot(phy2pixel(plotXs), phy2pixel(plotYs), "r-")

        # save or show the figure if necessary
        if saveToDir != "":
            plt.savefig(saveToDir)
        if showFig:
            plt.show()
        plt.close(figInit)
        plt.close(fig)

    def view_surface_density(
        self,
        coordinates,
        masses,
        size=20,
        binNum=100,
        ratio=1,
        vmin=None,
        vmax=None,
        tickNum1=9,
        tickNum2=9,
        cmap="jet",
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
            index = np.where(mat <= 0)
            mat[index] = 1
            mat = np.log10(mat)
            mat[index] = None
            return mat

        area = ((size - -size) / binNum) ** 2

        # x-y image
        imageXY = bin2d(
            x=coordinates[:, 1],
            y=coordinates[:, 0],
            values=masses,
            range=[[-size, size], [-size, size]],
            bins=binNum,
            statistic="sum",
        )[0]
        imageXY = logNormMat(imageXY / area)
        axFace.imshow(
            imageXY,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            interpolation=interpolation,
        )

        # x-z image
        imageXZ = bin2d(
            x=coordinates[:, 2],
            y=coordinates[:, 0],
            values=masses,
            range=[[-size * ratio, size * ratio], [-size, size]],
            bins=[int(binNum * ratio), binNum],
            statistic="sum",
        )[0]
        imageXZ = logNormMat(imageXZ / area)
        axLower.imshow(
            imageXZ,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            interpolation=interpolation,
        )

        # y-z image
        imageYZ = bin2d(
            x=coordinates[:, 1],
            y=coordinates[:, 2],
            values=masses,
            range=[[-size, size], [-size * ratio, size * ratio]],
            bins=[binNum, int(binNum * ratio)],
            statistic="sum",
        )[0]
        imageYZ = logNormMat(imageYZ / area)
        im = axRighter.imshow(
            imageYZ,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
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
        cmap="jet",
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
            cmap=cmap,
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
            cmap=cmap,
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
            cmap=cmap,
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

    def BPXstrength(self, Zs, masses=[]):
        """
        Calculate the BPX strength parameter.
        """
        if len(masses) == 0:
            denominator = len(Zs)
            numerator = np.sum(Zs**2)
        else:
            denominator = np.sum(masses)
            numerator = np.sum(masses * Zs**2)
        return numerator / denominator

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

    def monotonize_bar_angle(self, bar_angles, threshold=np.deg2rad(90)):
        """
        Monotonization of the bar angles (m=2 phase) based on its global increasing or decreasing trend.
        """
        # make sure that the bar angles are calculated in rads as the common fassion
        assert bar_angles.min() >= -np.pi / 2 and bar_angles.max() <= np.pi / 2
        sign = np.sign(np.mean(np.sign(bar_angles[1:] - bar_angles[:-1])))
        plain = bar_angles * 1  # avoid the overlap of values
        for i in range(len(bar_angles) - 1):
            if (plain[i + 1] - plain[i]) / sign <= -threshold:
                plain[i + 1 :] += np.pi * sign
        return plain

    def A2profile(
        self, phis, Rs, masses=[], Rmin=0.1, Rmax=20, RbinNum=40, normalize=True
    ):
        """
        Calculate the radial profile of the m=2 amplitudes.
        """
        RbinEdges = np.linspace(Rmin, Rmax, RbinNum + 1)
        A2s = []
        for i in range(RbinNum):
            index = np.where((Rs >= RbinEdges[i]) & (Rs < RbinEdges[i + 1]))[0]
            A2s.append(self.A2(phis=phis[index], masses=masses, normalize=normalize))
        return np.array(A2s), (RbinEdges[1:] + RbinEdges[:-1]) / 2

    def RbarThreshold(
        self, phis, Rs, masses=[], Rmin=0.01, Rmax=20, RbinNum=40, threshold=1
    ):
        """
        Calculate the bar radius that A2 reach some threshold, 1 for max.
        """
        # calculate the A2 profile
        profile = self.A2profile(
            phis=phis, Rs=Rs, masses=masses, Rmin=Rmin, Rmax=Rmax, RbinNum=RbinNum
        )[0]
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
        oCoordinates = np.column_stack((Rs, phi, coordinates[:, 2]))
        oVeloicties = np.column_stack((VR, Vphi, velocities[:, 2]))
        return oCoordinates, oVeloicties

    def anisotropy(self, Vrs, Vphis, Vthetas):
        """
        Calculate the radial anisotropy parameter from the spherical velocities.
        Return: the float for the anisotropy parameter.
        """
        sigmaR = np.nanstd(Vrs)
        sigmaPhi = np.nanstd(Vphis)
        sigmaTheta = np.nanstd(Vthetas)
        beta = 1 - (sigmaTheta**2 + sigmaPhi**2) / (2 * sigmaR**2)
        return beta

    def getRdHz(
        self,
        masses,
        cartesianCoordinates,
        Rmin=0.1,
        Rmax=30,
        RbinNum=60,
        Zmax=3,
        ZbinNum=18,
    ):
        """
        Calculate the disk scale length and height.
        """
        Rs = np.linalg.norm(cartesianCoordinates[:, :2], axis=1, ord=2)
        Zs = cartesianCoordinates[:, 2]
        Zmin = -abs(Zmax)
        RbinEdges = np.linspace(Rmin, Rmax, RbinNum + 1)
        ZbinEdges = np.linspace(Zmin, Zmax, ZbinNum + 1)
        Mtot = bin2d(Rs, Zs, masses, bins=[RbinEdges, ZbinEdges], statistic="sum")[0]

        def bin_center(data):
            return (data[1:] + data[:-1]) / 2

        # Transform it into \rho(R, z)
        deltaR = (Rmax - Rmin) / RbinNum
        deltaZ = (Zmax - Zmin) / ZbinNum
        Rcenters = bin_center(RbinEdges)
        Zcenters = bin_center(ZbinEdges)
        rhos = 1 * Mtot
        for i in range(rhos.shape[1]):
            weights = deltaR * deltaZ * np.pi * 2 * Rcenters
            rhos[:, i] /= weights
        index = np.where(rhos[:, int(len(Zcenters) / 2)] > 0)[0]
        res = linregress(
            Rcenters[index], np.log(rhos[:, int(len(Zcenters) / 2)][index])
        )
        Rd = -1 / res.slope
        index = np.where(rhos[0] > 0)[0]
        res = linregress(np.abs(Zcenters[index]), np.log(rhos[0][index]))
        Zd = -1 / res.slope

        return Rd, Zd
