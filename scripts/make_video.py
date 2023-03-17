import os
from subprocess import DEVNULL, STDOUT, check_call

path_to_files = 'designs/'
outdir = 'pngs/'
pdbs = [path_to_files + file for file in os.listdir(path_to_files)]
pdbs = sorted(pdbs, key=lambda f: int(f.split('/')[-1].split('_design')[0]))

designs = {}
for pdb in pdbs:
    traj = (pdb.split('_design_')[-1][:-4])
    if traj not in designs.keys():
        designs[traj] = [pdb]
    else:
        designs[traj].append(pdb)

images = {}
for traj in designs.keys():

    for i, pdb in enumerate(designs[traj]):
        if i == 0:
            # only open this one file without alignemt
            struc = pdb
            png = outdir + pdb.split('/')[-1][:-4] + '.png'
            lines = [
                f'load {struc}, struc',
                'spectrum b, blue_white_red, minimum=20, maximum=50',
                'bg_color white',
                'set ray_opaque_background, off',
                'set ray_trace_fog, 0.5',
                'set ray_shadows, off',
                'set ambient, 0.2',
                'ray 1200, 1200',
                f'png {png}',
                'quit'
            ]
            with open('make_png.pml', 'w') as f:
                for line in lines:
                    f.writelines(line + '\n')

            images[traj] = [png]
            print('hello')
            check_call(['pymol', '-cpihq', 'make_png.pml'], stdout=DEVNULL, stderr=STDOUT)
        else:
            # align to previous file
            struc = pdb
            prev_struc = designs[traj][i - 1]
            png = outdir + pdb.split('/')[-1][:-4] + '.png'
            lines = [
                f'load {struc}, struc',
                f'load {prev_struc}, prev_struc',
                'align struc, prev_struc',
                'delete prev_struc',
                'spectrum b, blue_white_red, minimum=20, maximum=50',
                'bg_color white',
                'set ray_opaque_background, off',
                'set ray_trace_fog, 0.5',
                'set ray_shadows, off',
                'set ambient, 0.2',
                'ray 1200, 1200',
                f'png {png}',
                'quit'
            ]
            with open('make_png.pml', 'w') as f:
                for line in lines:
                    f.writelines(line + '\n')

            images[traj].append(png)
            print('hello')
            check_call(['pymol', '-cpihq', 'make_png.pml'], stdout=DEVNULL, stderr=STDOUT)

import imageio.v2 as imageio

if not os.path.exists(outdir):
    os.mkdir(outdir)

# Read each PNG file and append to the list of frames
for i in images.keys():
    frames = []
    for file in images[i]:
        frames.append(imageio.imread(file))

    # Save the frames as a GIF
    output_file = f"design_{i}.gif"
    imageio.mimsave(output_file, frames, duration=0.1)