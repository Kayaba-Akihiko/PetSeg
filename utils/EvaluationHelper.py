#  Copyright (c) 2021-2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.
#
import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
from abc import ABC


class EvaluationHelper(ABC):

    @classmethod
    def dc_and_assd(cls, result: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
        dc = EvaluationHelper.dc(result, reference)
        try:
            assd = EvaluationHelper.assd(result, reference)
        except RuntimeError:
            # In case of all-zero sample
            result[-1, -1] = 1
            reference[0, 0] = 1
            assd = EvaluationHelper.assd(result, reference)
        return dc, assd

    @staticmethod
    def dc(result: np.ndarray, reference: np.ndarray) -> float:
        r"""
        https://loli.github.io/medpy/_modules/medpy/metric/binary.html
        Dice coefficient

        Computes the Dice coefficient (also known as Sorensen index) between the binary
        objects in two images.

        The metric is defined as

        .. math::

            DC=\frac{2|A\cap B|}{|A|+|B|}

        , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).

        Parameters
        ----------
        result : array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
        reference : array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.

        Returns
        -------
        dc : float
            The Dice coefficient between the object(s) in ```result``` and the
            object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).

        Notes
        -----
        This is a real metric. The binary images can therefore be supplied in any order.
        """
        result = np.atleast_1d(result.astype(bool))
        reference = np.atleast_1d(reference.astype(bool))

        intersection = np.count_nonzero(result & reference)

        size_i1 = np.count_nonzero(result)
        size_i2 = np.count_nonzero(reference)

        try:
            dc = 2. * intersection / float(size_i1 + size_i2)
        except ZeroDivisionError:
            dc = 0.0

        return dc

    @classmethod
    def assd(cls,
             result: np.ndarray,
             reference: np.ndarray,
             voxelspacing=None,
             connectivity=1) -> float:
        """
        https://loli.github.io/medpy/_modules/medpy/metric/binary.html
        Average symmetric surface distance.

        Computes the average symmetric surface distance (ASD) between the binary objects in
        two images.

        Parameters
        ----------
        result : array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
        reference : array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
        voxelspacing : float or sequence of floats, optional
            The voxelspacing in a distance unit i.e. spacing of elements
            along each dimension. If a sequence, must be of length equal to
            the input rank; if a single number, this is used for all axes. If
            not specified, a grid spacing of unity is implied.
        connectivity : int
            The neighbourhood/connectivity considered when determining the surface
            of the binary objects. This value is passed to
            `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
            The decision on the connectivity is important, as it can influence the results
            strongly. If in doubt, leave it as it is.

        Returns
        -------
        assd : float
            The average symmetric surface distance between the object(s) in ``result`` and the
            object(s) in ``reference``. The distance unit is the same as for the spacing of
            elements along each dimension, which is usually given in mm.

        See also
        --------
        :func:`asd`
        :func:`hd`

        Notes
        -----
        This is a real metric, obtained by calling and averaging

        >>> asd(result, reference)

        and

        >>> asd(reference, result)

        The binary images can therefore be supplied in any order.
        """
        assd = np.mean(
            (cls.asd(result, reference, voxelspacing, connectivity),
             cls.asd(reference, result, voxelspacing, connectivity)))
        return assd

    @classmethod
    def asd(cls, result, reference, voxelspacing=None, connectivity=1):
        """
        https://loli.github.io/medpy/_modules/medpy/metric/binary.html
        Average surface distance metric.

        Computes the average surface distance (ASD) between the binary objects in two images.

        Parameters
        ----------
        result : array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
        reference : array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
        voxelspacing : float or sequence of floats, optional
            The voxelspacing in a distance unit i.e. spacing of elements
            along each dimension. If a sequence, must be of length equal to
            the input rank; if a single number, this is used for all axes. If
            not specified, a grid spacing of unity is implied.
        connectivity : int
            The neighbourhood/connectivity considered when determining the surface
            of the binary objects. This value is passed to
            `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
            The decision on the connectivity is important, as it can influence the results
            strongly. If in doubt, leave it as it is.

        Returns
        -------
        asd : float
            The average surface distance between the object(s) in ``result`` and the
            object(s) in ``reference``. The distance unit is the same as for the spacing
            of elements along each dimension, which is usually given in mm.

        See also
        --------
        :func:`assd`
        :func:`hd`


        Notes
        -----
        This is not a real metric, as it is directed. See `assd` for a real metric of this.

        The method is implemented making use of distance images and simple binary morphology
        to achieve high computational speed.

        Examples
        --------
        The `connectivity` determines what pixels/voxels are considered the surface of a
        binary object. Take the following binary image showing a cross

        >>> from scipy.ndimage.morphology import generate_binary_structure
        >>> cross = generate_binary_structure(2, 1)
        array([[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]])

        With `connectivity` set to `1` a 4-neighbourhood is considered when determining the
        object surface, resulting in the surface

        .. code-block:: python

            array([[0, 1, 0],
                   [1, 0, 1],
                   [0, 1, 0]])

        Changing `connectivity` to `2`, a 8-neighbourhood is considered and we get:

        .. code-block:: python

            array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])

        , as a diagonal connection does no longer qualifies as valid object surface.

        This influences the  results `asd` returns. Imagine we want to compute the surface
        distance of our cross to a cube-like object:

        >>> cube = generate_binary_structure(2, 1)
        array([[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]])

        , which surface is, independent of the `connectivity` value set, always

        .. code-block:: python

            array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

        Using a `connectivity` of `1` we get

        >>> asd(cross, cube, connectivity=1)
        0.0

        while a value of `2` returns us

        >>> asd(cross, cube, connectivity=2)
        0.20000000000000001

        due to the center of the cross being considered surface as well.

        """
        sds = cls.__surface_distances(result, reference, voxelspacing, connectivity)
        asd = sds.mean()
        return asd

    @staticmethod
    def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
        """
        https://loli.github.io/medpy/_modules/medpy/metric/binary.html
        The distances between the surface voxel of binary objects in result and their
        nearest partner surface voxel of a binary object in reference.
        """
        result = np.atleast_1d(result.astype(bool))
        reference = np.atleast_1d(reference.astype(bool))
        if voxelspacing is not None:
            voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
            voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
            if not voxelspacing.flags.contiguous:
                voxelspacing = voxelspacing.copy()

        # binary structure
        footprint = generate_binary_structure(result.ndim, connectivity)

        # test for emptiness
        if 0 == np.count_nonzero(result):
            raise RuntimeError('The first supplied array does not contain any binary object.')
        if 0 == np.count_nonzero(reference):
            raise RuntimeError('The second supplied array does not contain any binary object.')

            # extract only 1-pixel border line of objects
        result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
        reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

        # compute average surface distance
        # Note: scipys distance transform is calculated only inside the borders of the
        #       foreground objects, therefore the input has to be reversed
        dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
        sds = dt[result_border]

        return sds
