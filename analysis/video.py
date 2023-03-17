import os
from subprocess import DEVNULL, STDOUT, check_call
import imageio.v2 as imageio
import tempfile

def make(pdb_path, png_path, video_path):
    pdbs = [os.path.join(pdb_path, file) for file in os.listdir(pdb_path)]
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
                png = os.path.join(png_path, pdb.split('/')[-1][:-4] + '.png')
                lines = [
                    f'load {struc}, struc',
                    'spectrum b, red_white_blue',
                    'bg_color white',
                    'set ray_opaque_background, off',
                    'set ray_trace_fog, 0.5',
                    'set ray_shadows, off',
                    'set ambient, 0.2',
                    'ray 1200, 1200',
                    f'png {png}',
                    'quit'
                ]
                # create a temporary file
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                    # write the contents of `lines` to the temporary file
                    for line in lines:
                        f.write(line + '\n')
                    temp_file_path = f.name

                # run PyMOL with the temporary file as input
                check_call(['pymol', '-cpihq', temp_file_path], stdout=DEVNULL, stderr=STDOUT)

            else:
                # align to previous file
                struc = pdb
                prev_struc = designs[traj][i - 1]
                png = os.path.join(png_path, pdb.split('/')[-1][:-4] + '.png')
                lines = [
                    f'load {struc}, struc',
                    f'load {prev_struc}, prev_struc',
                    'align struc, prev_struc',
                    'delete prev_struc',
                    'spectrum b, red_white_blue',
                    'bg_color white',
                    'set ray_opaque_background, off',
                    'set ray_trace_fog, 0.5',
                    'set ray_shadows, off',
                    'set ambient, 0.2',
                    'ray 1200, 1200',
                    f'png {png}',
                    'quit'
                ]
                # create a temporary file
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                    # write the contents of `lines` to the temporary file
                    for line in lines:
                        f.write(line + '\n')
                    temp_file_path = f.name

                # run PyMOL with the temporary file as input
                check_call(['pymol', '-cpihq', temp_file_path], stdout=DEVNULL, stderr=STDOUT)

    if not os.path.exists(png_path):
        os.mkdir(png_path)

    # Read each PNG file and append to the list of frames
    for i in images.keys():
        frames = []
        for file in images[i]:
            frames.append(imageio.imread(file))

        # Save the frames as a GIF
        output_file = os.path.join(video_path, f"design_{i}.gif")
        imageio.mimsave(output_file, frames, duration=0.1)