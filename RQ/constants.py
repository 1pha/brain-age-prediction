from pathlib import Path
from nilearn.datasets import load_mni152_brain_mask


mni_template = load_mni152_brain_mask()
MNI_SHAPE = mni_template.get_fdata().shape
MNI_AFFINE = mni_template.affine


BASE = Path.home() / "codespace/brain-age-prediction"
META_DIR = BASE / "meta_brain"

_WEIGHTS = BASE / "meta_brain" / "weights"
WEIGHT_DIR = _WEIGHTS / "default"

ANALYSIS_DIR = META_DIR / "analysis"
FS_DIR = ANALYSIS_DIR / "fastsurfer"
VBM_DIR = ANALYSIS_DIR / "vbm"

WEIGHTS_LIST = ["convnext-base-42",
                "convnext-base-43",
                "convnext-base-44",
                "convnext-tiny-42",
                "convnext-tiny-43",
                "convnext-tiny-44",
                "resnet10-42",
                "resnet10-43",
                "resnet10-44",
                "resnet18-42",
                "resnet18-43",
                "resnet18-44",
                "resnet34-42",
                "resnet34-43",
                "resnet34-44"]

MODEL_KEY = {"resnet10": "ResNet10",
             "resnet18": "ResNet18",
             "resnet34": "ResNet34",
             "convnext-tiny": "ConvNext-Tiny",
             "convnext-base": "ConvNext-Base",}

MODELS = ["resnet10", "resnet18", "resnet34", "convnext-tiny", "convnext-base"]
XAI_METHODS = ["gradxinput", "gcam_avg", "gbp", "smooth_gbp",
               "ggcam", "ggcam_avg", "deeplift", "ig"]
XAI_METHODS_MAPPER = {
    "gradxinput": "GradXInput",
    "gcam_avg": "GradCAM Avg.",
    "gbp": "GuidedBackprop",
    "smooth_gbp": "SmoothGBP",
    "ggcam": "Guided-GradCAM",
    "ggcam_avg": "GuidedGCAM Avg.",
    "deeplift": "DeepLIFT",
    "ig": "Integ. Gradients",
}

NUM_TEST = 3029

# Constants for plottings
XCOL, YCOL, HUECOL = "XAI Method", "Similarity", "Similarity Method"
