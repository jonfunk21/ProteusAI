import esm
from biotite.structure.io.pdb import PDBFile

fpath = '../example_data/structures/ASMT.pdb'

def get_representation(fpath):

    coords = PDBFile.read(fpath).get_coord()

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()

    rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)

    return rep

print(get_representation(fpath))