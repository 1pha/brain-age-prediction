import sys
sys.path.append('../')
try: 
    from models.model_util import load_model
    from data.data_util import DatasetPlus
except:
    from ..models.model_util import load_model
    from ..data.data_util import DatasetPlus

from .cams import *
from .smoothgrad import *
from .auggrad import *
from .visual_utils import plot_vismap, convert2nifti, exp_parser, brain_parser

class VisTool:

    __version = 'Apr 7. 2021'
    CAMS = {
        'gcam': CAM,
        'sgrad': SmoothGrad,
        'agrad': AugGrad,
    }

    def __init__(self, cfg, model, cam_type='agrad', **kwargs):

        '''
        cfg:
            Configuration dict. necessary
        model:
            PyTorch Model. Doesn't need to be trained (Better to give pretrained).
            You can pass a pretrained weights through load_weight method
        cam_type:
            type in which cam to use. 3 options for Apr 7. - gcam, sgrad, agrad.
            which stands for GradCAM, SmoothGrad, AugGrad respectively
        '''

        self.cfg = cfg
        self.model = model
        self.cam_type = cam_type
        self.vis_tool = VisTool.CAMS[self.cam_type](cfg, model, **kwargs)

    def load_weight(self, pth):

        '''
        Load pretraind weights to model. Use .pth file to load
        '''

        try:
            self.model.load_state_dict(torch.load(pth))
            print("Weights successfully loaded!")

        except:
            print("Error occurred during loading weights.")

    def run_vistool(self, x, y, visualize=False, title=None, **kwargs):

        self.model.to(self.cfg.device)
        x, y = x.to(self.cfg.device), y.to(self.cfg.device)
        vismap = self.vis_tool(x, y, **kwargs) # Should return (1, 96, 96, 96) visualization map

        if visualize:
            plot_vismap(brain=x, vismap=vismap, title=title)

        return vismap

    def run_pretrains_single(self, path, x, y, slc=None, visualize=True, **kwargs):

        '''
        Put path that contains all the .pth during training
        This will run and visualize all the .pth
        slc: Selecting a slice to view
        '''

        saved_models = sorted(glob(path), \
                              key=lambda x: int(x.split('ep')[1].split('-')[0]))
        vismaps = list()
        for idx, pth in enumerate(saved_models):

            # _, epoch = exp_parser(pth)
            self.load_weight(pth)
            vismaps.append(self.run_vistool(x, y, visualize=visualize, **kwargs))

        return vismaps

    def run_pretrains_dataloader(self, path, dataloader, slc=None, visualize=False, **kwargs):

        saved_models = sorted(glob(path), key=lambda x: int(x.split('ep')[1].split('-')[0]))
        
        vismap_ts = list()
        for idx, pth in enumerate(saved_models):

            print(f"{idx}th Pretrained")
            self.load_weight(pth)
            for i, (x, y) in enumerate(dataloader):

                if i % 10 == 0:
                    print(f"{i / len(dataloader) * 100:.3f}% DONE.")

                if i == 0:
                    vismap = self.run_vistool(x, y, visualize=visualize, **kwargs)
                
                else:
                    vismap += self.run_vistool(x, y, visualize=visualize, **kwargs)
            
            vismap_ts.append(vismap)

        return vismap_ts