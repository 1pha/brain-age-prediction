from glob import glob
import argparse

from registration import *


parser = argparse.ArgumentParser(description='Registration work')
parser.add_argument('--template', '-t', type=int, default=None,
                    help='Which model to use, default=vanilla')
parser.add_argument('--start', '-s', type=int, default=None,
                    help='Where to start registration')
parser.add_argument('--end', '-e', type=int, default=None,
                    help='Where to end registration')
args = parser.parse_args()


if __name__=="__main__":

    data_files = glob('../../../brainmask_nii/*.nii')
    data_files.sort()

    template = data_files.pop(args.template) if args.template else data_files.pop()
    print(f'Using {template} as a template.')

    for i, moving in enumerate(data_files[args.start:args.end]):

        fname = moving.split('\\')[-1].split('.')[0]
        print(f"{fname}")
        affreg = Registration(template=template, moving=moving).optimize()
        warped = affreg.transform(nib.load(moving).get_fdata())
        np.save(f'../../../brainmask_tlrc/{fname}_tlrc.npy', warped)