from pathlib import Path


BASE = Path.home() / "codespace/brain-age-prediction"
META_DIR = BASE / "meta_brain"

_WEIGHTS = BASE / "meta_brain" / "weights"
WEIGHT_DIR = _WEIGHTS / "default"

ANALYSIS_DIR = META_DIR / "analysis"
FS_DIR = ANALYSIS_DIR / "fastsurfer"
VBM_DIR = ANALYSIS_DIR / "vbm"

WEIGHTS_LIST = ["convnext-base-42",
                "convnext-base-43",
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

MODELS = ["resnet10", "resnet18", "resnet34", "convnext-tiny", "convnext-base"]
XAI_METHODS = ["gradxinput", "gcam_avg", "gbp", "ggcam", "deconv", "deeplift", "ig"]