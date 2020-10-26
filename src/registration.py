import os
from glob import glob
import numpy as np
np.set_printoptions(precision=4, suppress=True)

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'

import nibabel as nib

from dipy.viz import regtools
from dipy.align.imaffine import (AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric

if __name__=="__main__":
    
    data_files = list(filter(lambda x: x.split('\\')[-1][:3] == 'IXI', glob('../../../brainmask_nii/*.nii')))
    template_img = nib.load(data_files.pop())
    template_data, template_affine = template_img.get_fdata(), template_img.affine

    moving_img = nib.load(data_files[0])
    moving_data, moving_affine = moving_img.get_fdata(), moving_img.affine

    # 00. Set Params
    # The mismatch metric
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    # The optimization strategy
    level_iters = [10, 10, 5]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    # 01. Define Affine Registration
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    # 02. Registration
    # 02-1. Translation
    transform = TranslationTransform3D()
    params0 = None
    translation = affreg.optimize(template_data, moving_data, transform, params0,
                                template_affine, moving_affine)

    # 02-2. Rigid
    transform = RigidTransform3D()
    rigid = affreg.optimize(template_data, moving_data, transform, params0,
                            template_affine, moving_affine,
                            starting_affine=translation.affine)

    # 02-3. Affine
    transform = AffineTransform3D()
    # Bump up the iterations to get an more exact fit
    affreg.level_iters = [1000, 1000, 100]
    affine = affreg.optimize(template_data, moving_data, transform, params0,
                            template_affine, moving_affine,
                            starting_affine=rigid.affine)
    
    # 02-4. SymmetricDiffeomorphicRegistration
    # The mismatch metric
    metric = CCMetric(3)
    # The optimization strategy:
    level_iters = [10, 10, 5]
    # Registration object
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
    mapping = sdr.optimize(template_data, moving_data, template_affine,
                            moving_affine, affine.affine)

    # Visualization
    warped_moving = mapping.transform(moving_data)
    regtools.overlay_slices(template_data, warped_moving, None, 0,
                            "Template", "Transformed")
    regtools.overlay_slices(template_data, warped_moving, None, 1,
                            "Template", "Transformed")
    regtools.overlay_slices(template_data, warped_moving, None, 2,
                            "Template", "Transformed")
