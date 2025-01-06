from pathlib import Path
from nilearn.datasets import load_mni152_brain_mask


mni_template = load_mni152_brain_mask()
MNI_SHAPE = mni_template.get_fdata().shape
MNI_AFFINE = mni_template.affine


BASE = Path.home() / "codespace/brain-age-prediction"
MEDIA_ASSET_DIR = BASE / "RQ" / "media_assets"
ASSET_DIR = BASE / "RQ" / "assets"
BIOBANK = BASE / "biobank"
TESTFILE = BIOBANK / "ukb_test_age_exist240221.csv"
META_DIR = BASE / "meta_brain"

_WEIGHTS = BASE / "meta_brain" / "weights"
WEIGHT_DIR = _WEIGHTS / "default"

ANALYSIS_DIR = META_DIR / "analysis"
FS_DIR = ANALYSIS_DIR / "fastsurfer"
VBM_DIR = ANALYSIS_DIR / "vbm"
OCC_DIR = ANALYSIS_DIR / "occlusion"

MODEL_KEY = {"resnet10": "ResNet10",
             "resnet18": "ResNet18",
             "resnet34": "ResNet34",
             "convnext-tiny": "ConvNeXt-Tiny",
             "convnext-small": "ConvNeXt-Small",
             "convnext-base": "ConvNeXt-Base",
             "densenet121": "Densenet121",
             "densenet169": "Densenet169",
             "densenet264d": "Densenet264d",
             }

MODELS = ["resnet10", "resnet18", "resnet34",
          "convnext-tiny", "convnext-small", "convnext-base",
          "densenet121", "densenet169", "densenet264d"]
MODELS_SIZE = {
    "resnet10": "3.6M",
    "resnet18": "8.3M",
    "resnet34": "15.8M",
    "convnext-tiny": "3.3M",
    "convnext-small": "12.5M",
    "convnext-base": "31.3M",
    "densenet121": "11.2M",
    "densenet169": "18.5M",
    "densenet264d": "90.4M",
}

XAI_METHODS = ["gradxinput", "gcam_avg", "gbp", "smooth_gbp",
               "ggcam", "ggcam_avg", "deeplift", "ig"]
XAI_METHODS_MAPPER = {
    "gradxinput": "GradXInput",
    "gcam_avg": "GradCAM",
    "gbp": "GuidedBackprop.",
    "smooth_gbp": "SmoothGBP",
    "ggcam": "Guided-GCAM",
    "ggcam_avg": "GGCAM Avg.",
    "deeplift": "DeepLIFT",
    "ig": "Integ. Gradients",
}

SEEDS = range(42, 52)
NUM_TEST = 3029

# Constants for plottings
XCOL, YCOL, HUECOL = "XAI Method", "Similarity", "Similarity Method"

PERF_AVG = dict(mae=2.55094, r2=0.822, mse=10.3914)
PERF_AVG_ADNI = dict(acc=0.8437, f1=0.7759, auroc=0.91331)

UKB_FEW_THD = 4.950344
ADNI_FWE_THD = 5.071649

ROI_COLUMNS = [
    '3rd-Ventricle',
 '4th-Ventricle',
 'Brain-Stem',
#  'CC_Anterior',
#  'CC_Central',
#  'CC_Mid_Anterior',
#  'CC_Mid_Posterior',
#  'CC_Posterior',
 'CSF',
 'Left-Accumbens-area',
 'Left-Amygdala',
 'Left-Caudate',
 'Left-Cerebellum-Cortex',
 'Left-Cerebellum-White-Matter',
 'Left-Cerebral-White-Matter',
 'Left-Hippocampus',
 'Left-Inf-Lat-Vent',
 'Left-Lateral-Ventricle',
 'Left-Pallidum',
 'Left-Putamen',
 'Left-Thalamus',
 'Left-VentralDC',
 'Left-choroid-plexus',
 'Right-Accumbens-area',
 'Right-Amygdala',
 'Right-Caudate',
 'Right-Cerebellum-Cortex',
 'Right-Cerebellum-White-Matter',
 'Right-Cerebral-White-Matter',
 'Right-Hippocampus',
 'Right-Inf-Lat-Vent',
 'Right-Lateral-Ventricle',
 'Right-Pallidum',
 'Right-Putamen',
 'Right-Thalamus',
 'Right-VentralDC',
 'Right-choroid-plexus',
 'WM-hypointensities',
 'ctx-lh-caudalanteriorcingulate',
 'ctx-lh-caudalmiddlefrontal',
 'ctx-lh-cuneus',
 'ctx-lh-entorhinal',
 'ctx-lh-fusiform',
 'ctx-lh-inferiorparietal',
 'ctx-lh-inferiortemporal',
 'ctx-lh-insula',
 'ctx-lh-isthmuscingulate',
 'ctx-lh-lateraloccipital',
 'ctx-lh-lateralorbitofrontal',
 'ctx-lh-lingual',
 'ctx-lh-medialorbitofrontal',
 'ctx-lh-middletemporal',
 'ctx-lh-paracentral',
 'ctx-lh-parahippocampal',
 'ctx-lh-parsopercularis',
 'ctx-lh-parsorbitalis',
 'ctx-lh-parstriangularis',
 'ctx-lh-pericalcarine',
 'ctx-lh-postcentral',
 'ctx-lh-posteriorcingulate',
 'ctx-lh-precentral',
 'ctx-lh-precuneus',
 'ctx-lh-rostralanteriorcingulate',
 'ctx-lh-rostralmiddlefrontal',
 'ctx-lh-superiorfrontal',
 'ctx-lh-superiorparietal',
 'ctx-lh-superiortemporal',
 'ctx-lh-supramarginal',
 'ctx-lh-transversetemporal',
 'ctx-rh-caudalanteriorcingulate',
 'ctx-rh-caudalmiddlefrontal',
 'ctx-rh-cuneus',
 'ctx-rh-entorhinal',
 'ctx-rh-fusiform',
 'ctx-rh-inferiorparietal',
 'ctx-rh-inferiortemporal',
 'ctx-rh-insula',
 'ctx-rh-isthmuscingulate',
 'ctx-rh-lateraloccipital',
 'ctx-rh-lateralorbitofrontal',
 'ctx-rh-lingual',
 'ctx-rh-medialorbitofrontal',
 'ctx-rh-middletemporal',
 'ctx-rh-paracentral',
 'ctx-rh-parahippocampal',
 'ctx-rh-parsopercularis',
 'ctx-rh-parsorbitalis',
 'ctx-rh-parstriangularis',
 'ctx-rh-pericalcarine',
 'ctx-rh-postcentral',
 'ctx-rh-posteriorcingulate',
 'ctx-rh-precentral',
 'ctx-rh-precuneus',
 'ctx-rh-rostralanteriorcingulate',
 'ctx-rh-rostralmiddlefrontal',
 'ctx-rh-superiorfrontal',
 'ctx-rh-superiorparietal',
 'ctx-rh-superiortemporal',
 'ctx-rh-supramarginal',
 'ctx-rh-transversetemporal'
]