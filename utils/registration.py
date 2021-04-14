import os
import time
from glob import glob
import numpy as np
np.set_printoptions(precision=4, suppress=True)

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'

import nibabel as nib
from nilearn.datasets import load_mni152_template, load_mni152_brain_mask

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

def inform(original_fn):

    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print(f"[{original_fn.__name__.capitalize()}] {end_time-start_time:.1f} sec ")
        return result

    return wrapper_fn

class Registration:

    def __init__(self, moving, template=None, **kwargs):
        '''
        template, moving: fname
        '''

        if isinstance(template, str):
            template_img = nib.load(template)

        else:
            template_img = load_mni152_template()
        self.template_data, self.template_affine = template_img.get_fdata(), template_img.affine

        moving_img = nib.load(moving)
        self.moving_data, self.moving_affine = moving_img.get_fdata(), moving_img.affine

        # Set params
        self.nbins = 32
        self.sampling_prop = None
        self.metric = MutualInformationMetric(self.nbins, self.sampling_prop)

        self.level_iters = [10, 10, 5]
        self.sigmas = [3.0, 1.0, 0.0]
        self.factors = [4, 2, 1]
        self.params0 = None

        self.affreg = AffineRegistration(metric=self.metric,
                                        level_iters=self.level_iters,
                                        sigmas=self.sigmas,
                                        factors=self.factors)


    def optimize(self):

        translation = self.transformation()
        rigid = self.rigid(translation)
        affine = self.affine(rigid)
        sdr = self.symmetricdr(affine)
        return sdr


    @inform
    def transformation(self):
        transform = TranslationTransform3D()
        return self.affreg.optimize(self.template_data, self.moving_data, transform, self.params0,
                                    self.template_affine, self.moving_affine)

    @inform
    def rigid(self, translation):
        transform = RigidTransform3D()
        return self.affreg.optimize(self.template_data, self.moving_data, transform, self.params0,
                                self.template_affine, self.moving_affine,
                                starting_affine=translation.affine)

    @inform
    def affine(self, rigid):
        transform = AffineTransform3D()
        # Bump up the iterations to get an more exact fit
        self.affreg.level_iters = [1000, 1000, 100]
        return self.affreg.optimize(self.template_data, self.moving_data, transform, self.params0,
                                self.template_affine, self.moving_affine,
                                starting_affine=rigid.affine)

    @inform
    def symmetricdr(self, affine):
        # The mismatch metric
        metric = CCMetric(3)
        # The optimization strategy:
        level_iters = [10, 10, 5]
        # Registration object
        sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
        mapping = sdr.optimize(self.template_data, self.moving_data,
                                self.template_affine, self.moving_affine,
                                affine.affine)
        return mapping


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
    print("1. Translation Optimization")
    transform = TranslationTransform3D()
    params0 = None
    translation = affreg.optimize(template_data, moving_data, transform, params0,
                                template_affine, moving_affine)

    # 02-2. Rigid
    print("2. Rigid Optimization")
    transform = RigidTransform3D()
    rigid = affreg.optimize(template_data, moving_data, transform, params0,
                            template_affine, moving_affine,
                            starting_affine=translation.affine)

    # 02-3. Affine
    print("3. Affine Optimization")
    transform = AffineTransform3D()
    # Bump up the iterations to get an more exact fit
    affreg.level_iters = [1000, 1000, 100]
    affine = affreg.optimize(template_data, moving_data, transform, params0,
                            template_affine, moving_affine,
                            starting_affine=rigid.affine)
    
    # 02-4. SymmetricDiffeomorphicRegistration
    print("4. SDR Optimization")
    # The mismatch metric
    metric = CCMetric(3)
    # The optimization strategy:
    level_iters = [10, 10, 5]
    # Registration object
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
    mapping = sdr.optimize(template_data, moving_data, template_affine,
                            moving_affine, affine.affine)

    # Visualization
    print("+. Visualization")
    warped_moving = mapping.transform(moving_data)
    regtools.overlay_slices(template_data, warped_moving, None, 0,
                            "Template", "Transformed")
    regtools.overlay_slices(template_data, warped_moving, None, 1,
                            "Template", "Transformed")
    regtools.overlay_slices(template_data, warped_moving, None, 2,
                            "Template", "Transformed")
