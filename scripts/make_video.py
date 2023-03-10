import os
from subprocess import DEVNULL, STDOUT, check_call

path_to_files = 'hallucinations/'
outdir = 'pngs/'
pdbs = [path_to_files + file for file in os.listdir(path_to_files)]
pdbs = sorted(pdbs, key=lambda f: int(f.split('/')[-1].split('_hallucination')[0]))

hallucinations = {}
for pdb in pdbs:
    traj = (pdb.split('_hallucination_')[-1][:-4])
    if traj not in hallucinations.keys():
        hallucinations[traj] = [pdb]
    else:
        hallucinations[traj].append(pdb)

images = {}
for traj in hallucinations.keys():

    for i, pdb in enumerate(hallucinations[traj]):
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
            check_call(['pymol', 'make_png.pml'], stdout=DEVNULL, stderr=STDOUT)
        else:
            # align to previous file
            struc = pdb
            prev_struc = hallucinations[traj][i - 1]
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
            check_call(['pymol', '-cpihq', 'make_png.pml'], stdout=DEVNULL, stderr=STDOUT)

import imageio

# Read each PNG file and append to the list of frames
for i in images.keys():
    frames = []
    for file in images[i]:
        frames.append(imageio.imread(file))

    # Save the frames as a GIF
    output_file = f"hallucination_{i}.gif"
    imageio.mimsave(output_file, frames, duration=0.5)