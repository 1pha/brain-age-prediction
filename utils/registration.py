import os
import time
from glob import glob
import numpy as np
np.set_printoptions(precision=4, suppress=True)

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'

import yaml

import nibabel as nib
from nilearn.datasets import load_mni152_template

from dipy.viz import regtools
from dipy.align.imaffine import (AffineMap,
                                transform_centers_of_mass,
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
        print(f"[{original_fn.__name__}] {end_time-start_time:.1f} sec ")
        return result

    return wrapper_fn

class Registration:

    '''
    Look up https://bic-berkeley.github.io/psych-214-fall-2016/dipy_registration.html here
    for detailed explanations of usage.
    '''

    __version__ = '0.2'
    __date__ = 'Apr.17 2021'

    def __init__(self, template=None, cfg=None, **kwargs):
        '''
        # TODO:
        1. Average time, t-test, find out which is weird
        2. Add Explanation
        '''
        self.set_template(template)

        ### CONFIGURATION FILE ###
        if cfg is None:
            with open('utils/registration.yml', 'r') as y:
                cfg = yaml.load(y)

        # CONFIGURE PIPELINE
        self.pipeline = cfg['Pipeline']

        # CONFIGUATION LINEAR-TRANSFORMATION 
        self.lin_cfg = cfg['Linear']
        lin_metric_cfg = self.lin_cfg['metric']
        self.metric = eval(lin_metric_cfg['name'])(
            lin_metric_cfg['nbins'],
            lin_metric_cfg['sampling_prop']
        )

        self.default_cfg = self.lin_cfg['default']
        self.params0 = self.default_cfg['params0']
        self.lin_cfgs_dict = {
            'TranslationTransform3D': self.lin_cfg['translation'],
            'RigidTransform3D': self.lin_cfg['rigid'],
            'AffineTransform3D': self.lin_cfg['affine'], 
        }

        # INSTANTIATE AFFINE-REGISTRATION
        self.affreg = AffineRegistration(metric=self.metric)
        self.set_params(**self.default_cfg)

        # CONFIGUATION NONLINEAR-TRANSFORMATION
        self.nonlin_cfg = cfg['NonLinear']
        nonlin_metric_cfg = self.nonlin_cfg['metric']
        self.sdr_cfg = {k: v for k, v in self.nonlin_cfg['symmetricdr'].items()}
        self.sdr_cfg['metric'] = eval(nonlin_metric_cfg['name'])(nonlin_metric_cfg['dim'])

        # MISC
        self.opt_res = dict() # OPTIMIZATION RESLUTS(LOSS)
        self.steps = dict()   # TRANSFORMATION INSTANCES ABLES TO TRANSFORM BRAINS
        self.debug = cfg['Debug']['switch']

    def __call__(self, moving, template=None, **kwargs):

        if template is None:
            pass

        else:
            self.set_template(template)

        self.fname = moving.split('\\')[-1]
        print(f"Working on: {self.fname}")
        
        moving_img = nib.load(moving)
        self.moving_data, self.moving_affine = moving_img.get_fdata(), moving_img.affine
        self.optimize(**kwargs)

    def set_template(self, template):

        '''
        template file should be '.nii' extension files
        '''

        if isinstance(template, str):
            template_img = nib.load(template)

        else:
            template_img = load_mni152_template()

        self.static_data, self.static_affine = template_img.get_fdata(), template_img.affine

    def set_params(self, **kwargs):

        for key, value in kwargs.items():
            setattr(self.affreg, key, value)

    @inform
    def optimize(self, pipeline=None):

        '''
        # STEPS
        1. Transform of Centers of Mass # <newly added
        2. Linear(Affine) Registration
            2.1 Instantiate AffineRegistration class
            2.2 Translation
            2.3 Rigid
            2.4 Affine
        3. Non-Linear Registration
            3.1 Symmetric Diffeomorphic Registration (SDR)
        '''

        if pipeline is None:
            pipeline = self.pipeline

        self.opt_cfg = {
            'static': self.static_data,
            'static_grid2world': self.static_affine,
            'moving': self.moving_data,
            'moving_grid2world': self.moving_affine,
        }
        self.starting_affine = None

        for pipes in pipeline:

            if pipes == 'transform_centers_of_mass':
                self.c_of_mass = transform_centers_of_mass(**self.opt_cfg)
                self.starting_affine = self.c_of_mass.affine

                self.opt_cfg['params0'] = self.params0

            elif pipes == 'SymmetricDiffeomorphicRegistration':
                sdr = SymmetricDiffeomorphicRegistration(**self.sdr_cfg)
                self.steps[pipename] = sdr.optimize(self.static_data, self.moving_data,
                                        self.static_affine, self.moving_affine,
                                        affine.affine)

            else:
                self._linear_by_parts('TranslationTransform3D')

    def _linear_by_parts(self, pipename, ret_metric=True):

        _start_time = time.time()
        self.opt_cfg['transform'] = eval(pipename)()
        self.opt_cfg['starting_affine'] = self.starting_affine
        self.opt_cfg['ret_metric'] = ret_metric
        self.set_params(**self.lin_cfgs_dict[pipename])
        self.steps[pipename], _, self.opt_res[pipename] = self.affreg.optimize(**self.opt_cfg)
        self.starting_affine = self.steps[pipename].affine
        print(f'[{pipename}] {time.time() - _start_time:.1f} sec',
                end=f':: LOSS {self.opt_res[pipename]:.5f}\n' if self.debug else '\n')

    def visual_warped(self, step_name=None):

        def vis(step_name):
            resampled = self.steps[step_name].transform(self.moving_data)
            print(f'Step [{step_name}]')
            regtools.overlay_slices(self.static_data, resampled, None, 0, "Template", "Moving")
            regtools.overlay_slices(self.static_data, resampled, None, 1, "Template", "Moving")
            regtools.overlay_slices(self.static_data, resampled, None, 2, "Template", "Moving")
            
        if step_name == 'all':
            for step_name in self.steps.keys():
                vis(step_name)
        
        else:
            if step_name is None:
                step_name = 'sdr'
            
            vis(step_name)

    def save(self, root='../../../brainmask_mni/', fname=None, step_name=None, extension='npy'):

        if step_name is None:
            step_name = 'SymmetricDiffeomorphicRegistration'

        try:
            fpath = root + self.fname.split('nii')[0] + extension 

            if extension == 'npy':
                np.save(
                    f"{fpath}", # FILE NAME 
                    self.steps[step_name].transform(self.moving_data) # ARRAY
                )

            elif extension == 'nii':
                nib.Nifti1Image(
                    self.steps[step_name].transform(self.moving_data), # NIFTI IMAGE
                    f"{fpath}" # FILE NAME
                )
            
            print(f"Saved Successfully with fpath={fpath}")
        
        except:
            print("Failed Saving")
            raise

def compare_brains(left, right, left_title="A", right_title="B"):
    '''
    left, right brains should be in the form of np.ndarray with same dimensions
    '''

    regtools.overlay_slices(left, right, None, 0, left_title, right_title)
    regtools.overlay_slices(left, right, None, 1, left_title, right_title)
    regtools.overlay_slices(left, right, None, 2, left_title, right_title)


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

    c_of_mass = transform_centers_of_mass(template_data, template_imge, moving_data, moving_img)
    starting_affine = c_of_mass.affine

    # 02. Registration
    # 02-1. Translation
    print("1. Translation Optimization")
    transform = TranslationTransform3D()
    params0 = None
    translation = affreg.optimize(template_data, moving_data, transform, params0,
                                template_affine, moving_affine, starting_affine=starting_affine)

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
