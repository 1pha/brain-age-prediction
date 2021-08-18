from glob import glob
from IPython.display import clear_output

from .cams import *
from .smoothgrad import *
from .auggrad import *
from .visual_utils import plot_vismap


def deprecate(func):
    print(f'This {(func.__name__)} function is no longer supported since version=0.2')
    def class_wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return class_wrapper


def get_weight_dict(prefix):

    return {
        model_name: sorted(\
            glob(f'{prefix}/{model_name}/*.pt'), \
            key=lambda x: int(x.split('\\ep')[-1].split('_')[0]) \
        ) for model_name in ['encoder', 'regressor']
    }


class VisTool:

    __version__ = '0.2'
    __date__ = 'Aug 18. 2021'
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
        Load pretraind weights to model.
        Either use
            dict:
                that contains {model_name: weight_path}
            str:
                directly .pth
        '''

        try:
            if isinstance(pth, dict):
                self.model.load_weight(pth)

            elif isinstance(pth, str):
                self.model.load_state_dict(torch.load(pth))

            print("Weights successfully loaded!")

        except:
            print("An error occurred during loading weights.")
            raise


    def __call__(
        self,
        x=None,
        y=None,
        dataloader=None,
        visualize=False,
        slice_index=48,
        weight=None,
        prefix=None, # DEPRECATED
        layer_index=None,
    ):
        '''
        This method interacts with `trainer`.
        Arguments:
            x: torch.tensor
                A single brain image with 5D (batch=1, channel=1, Height, Width, Depth)
                If not 5D, function will automatically convert it to desired shape.
            y: torch.tensor
                A single age (or any target) in float contained in tensor

            dataloader: torch.utils.data.DataLoader
                Dataloader from torch that yields (x, y) pair.
                If dataloader returns x, y, d, this will be processed internally.
            average: bool, default=True # FURTHER IMPLEMENTED
                If dataloader is given, return an averaged visual map of all the brains
                contained inside the dataloader.

            *Note: One of (x, y) or dataloader should be given.

            visualize: bool
                Whehter to show saliency map on the prompt or not.
                If True, `plot_vistool` will be executed.
            title: str, default=None, optional # FURTHER IMPLEMENTED
                If given, this will be used as title during visualization
            slice_index: int|list, default=48
                Select which slice to visualize. Either integer or list of integers be given.

            weight: dict, default=None, optional
                Dictionary that constructed with {model_name: weight_path}.
                Model of attribute would be set with a given weight.
                Note that this should be a single checkpoint of a model.
                To use multiple model, please use `path` instead.
            path: dict, default=None <- DEPRECATED
                directory of path that contains a total checkpoint of single run.
                Directory must follow the next architecture
                    path/
                        -encoder
                        -regressor
                        -domainer (optional)
            
            layer_index: int|list default=None
                Select a layer/layers to retrieve a saliency map.
                If None is given, find all layers' visual map
                
        '''

        def run():

            if x is not None and y is not None: # (x, y) pair given
                if dataloader is not None:
                    print("Don't need to pass dataloader if x and y is given. "\
                        "This dataloader will be ignored.")
                    
                vismap = self.run_vistool(x, y, layer_index=layer_index)
                brain = x

            elif dataloader is not None: # dataloader given.
                if x is not None and y is not None:
                    print("(x, y) pair overpowers in priority against dataloader. "\
                        "VisMap with a single (x, y) pair will be returned")

                vismap = [np.zeros(self.cfg.resize) for l in range(len(self.vis_tool.conv_layers))]
                brain = torch.zeros((1, 1, *self.cfg.resize))
                for _x, _y, _ in dataloader:

                    _vismap = self.run_vistool(_x, _y, layer_index=layer_index)
                    brain += _x
                    for i, v in enumerate(_vismap):
                        vismap[i] += v

            if visualize:
                for idx, layer in enumerate(vismap):
                    plot_vismap(brain, layer, slc=slice_index, title=f"{idx}th layer.")

            return vismap

        if prefix is not None:
            weights = get_weight_dict(prefix)
            vismaps = list()
            for encoder_weight, regressor_weight in zip(weights['encoder'], weights['regressor']):
        
                self.load_weight({
                    'encoder': encoder_weight,
                    'regressor': regressor_weight,
                })
                vismaps.append(run())

        if weight is not None:
            self.load_weight(weight)
            return run()

        elif prefix is not None:

            weights = get_weight_dict(prefix)
            vismaps = list()
            for encoder_weight, regressor_weight in zip(weights['encoder'], weights['regressor']):
        
                self.load_weight({
                    'encoder': encoder_weight,
                    'regressor': regressor_weight,
                })
                vismaps.append(run())

            return vismaps # [VISMAP_EP1(=[LAYER1, LAYER2, ...]), VISMAP_EP2, ...]

        else:
            print("None of weight neither prefix is given.")
            return        

        
    def run_vistool(self, x, y, layer_index=None, **kwargs):

        self.model.to(self.cfg.device)
        x, y = x.to(self.cfg.device), y.to(self.cfg.device)
        vismap = self.vis_tool(x, y, layer_index=layer_index, **kwargs) # Should return (1, 96, 96, 96) visualization map

        return vismap


    @deprecate
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


    @deprecate
    def run_dataloader(self, dataloader, pth=None, slc=None, visualize=False, **kwargs):
        '''
        Runs VisTool on a single Dataloader
        No need to give a speicific path if its already pretrained, but you can give it as a input 'pth'
        '''

        if pth is not None:
            self.load_weight(pth)

        print(f"{len(dataloader)} brains in total.")
        for i, (x, y) in enumerate(dataloader):

            if i % 10 == 0:
                print(f"{i / len(dataloader) * 100:.3f}% DONE.")

            if i == 0:
                avg_vismap = self.run_vistool(x, y, visualize=visualize, **kwargs)
            
            else:
                avg_vismap += self.run_vistool(x, y, visualize=visualize, **kwargs)

        avg_vismap = avg_vismap / len(dataloader)
        return avg_vismap


    @deprecate
    def run_pretrains_dataloader(self, path, dataloader, slc=None, visualize=False, checkpoint=True, **kwargs):

        saved_models = sorted(glob(path), key=lambda x: int(x.split('ep')[1].split('-')[0]))
        
        self.vismap_ts = list()
        for idx, pth in enumerate(saved_models):

            print(f"{idx}th Pretrained")
            self.load_weight(pth)
            print(f"PTH: {pth}")
            vismap = self.run_dataloader(dataloader, visualize=visualize, **kwargs)
            self.vismap_ts.append(vismap)
            if checkpoint:
                att_path = 'attention_maps'.join(path.split('models'))[:-6] + '_tmp'
                np.save(f'{att_path}/{str(idx).zfill(3)}.npy', v)
            clear_output()