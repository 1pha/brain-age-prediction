import os
import sys

from .misc import get_today


def _generate_name(model_args, data_args, training_args, misc_args):

    features = {
        "date": get_today(),
        "model_name": model_args.model_name,
        "seed": str(misc_args.seed),
    }
    dir_name = f"[{features['date']}]"
    dir_name += f"{features['model_name']}"
    dir_name += f"-{features['seed']}"
    return dir_name


def set_path(model_args, data_args, training_args, misc_args, logger):

    if misc_args.output_dir is None:
        misc_args.output_dir = _generate_name(
            model_args, data_args, training_args, misc_args
        )

    # Check if same name folder exists.
    output_fullpath = os.path.join(misc_args.output_path, misc_args.output_dir)
    if os.path.exists(output_fullpath):
        logger.warn(
            f"You're overwriting on a directory {output_fullpath}. Please check"
        )
        if misc_args.overwrite_output:
            logger.warn(
                f"Overwriting arguments checked to True. We will overwrite on the folder."
            )
            return output_fullpath

        else:
            logger.info(f"Overwriting arguments checked to False. Stop training.")
            return False
    else:
        os.makedirs(output_fullpath, exist_ok=True)
        logger.info(f"No errors found in setting path.")
        return output_fullpath
