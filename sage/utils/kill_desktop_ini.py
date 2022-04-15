import os


def kill_desktop_ini(root_dir):

    for (root, dirs, files) in os.walk(root_dir):

        if len(files) > 0:
            for file_name in files:
                if file_name.endswith(".ini"):
                    os.remove(f"{root}/{file_name}")


if __name__ == "__main__":

    kill_desktop_ini("../../brain_data/workspace/3d_brain/mlruns/1")
