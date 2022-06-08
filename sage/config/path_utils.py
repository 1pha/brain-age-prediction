import os
import sys

from ..utils.misc import get_today


def _generate_name(data_args, training_args, misc_args):

    features = {
        "date": get_today(),
        "model_name": training_args.model_name,
        "seed": str(misc_args.seed),
    }
    dir_name = f"[{features['date']}]"
    dir_name += f"{features['model_name']}"
    """ Temporary names for testing"""
    dir_name += f"-{features['seed']}"
    # dir_name += f"lr{training_args.learning_rate}"
    # dir_name += f"warmup{int(training_args.warmup_ratio * 100)}%"

    return dir_name, features


def set_path(data_args, training_args, misc_args):

    if misc_args.output_dir is None:
        misc_args.output_dir, features = _generate_name(
            data_args, training_args, misc_args
        )
        run_name = f'{features["model_name"]}-{features["seed"]}'
    else:
        run_name = misc_args.output_dir

    # Check if same name folder exists.
    output_fullpath = os.path.join(misc_args.output_path, misc_args.output_dir)
    if os.path.exists(output_fullpath):
        print(f"You're overwriting on a directory {output_fullpath}. Please check")
        if misc_args.overwrite_output:
            print(
                f"Overwriting arguments checked to True. We will overwrite on the folder."
            )
            return output_fullpath, run_name

        else:
            print(f"Overwriting arguments checked to False. Stop training.")
            raise
    else:
        os.makedirs(output_fullpath, exist_ok=True)
        print(f"No errors found in setting path.")
        return output_fullpath, run_name
