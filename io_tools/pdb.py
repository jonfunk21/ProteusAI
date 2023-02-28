def format_integer(num, length=4):
    """
    Formats an integer as a string with leading spaces to reach a fixed length.
    """
    num_str = str(num)
    if len(num_str) >= length:
        return num_str
    else:
        num_spaces = length - len(num_str)
        return ' ' * num_spaces + num_str


def clean_pdb(pdb, remove_chains: list = [], remove_water: bool = False, keep_hetat: bool = True,
              renumber_chains: bool = True, outfile: str = None):
    """
    Removes header and brings pdb file to a 'basic' format which is
    more likely accepted by Rosetta.

    Parameters:
        pdb (str): path to file
        remove_chains (list, optional): list of chains which should be removed. Default []
        remove_water (bool, optional): remove water molecules if True. Default False
        keep_hetat (bool, optional): removes lines starting with HETATM if False. Default True
        renumber_chains (bool, optional): renumbers residues, starting from 1 if True. Default True
        outfile (str, optional): saves output file at target location.

    Retruns:
        list: lines of cleaned pdb file
    """
    lines = []
    c = ''  # keeps track of chain
    with open(pdb, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or (line.startswith('HETATM') and keep_hetat):
                if line[21] != c:
                    c = line[21]
                    if renumber_chains:
                        const = 1 - int(line[22:26])
                    else:
                        const = 0
                line = line[:72] + ' ' + line[73:78] + '\n'
                chain = line[21]
                num = format_integer(int(line[22:26]) + const)
                res = line[17:20]
                if chain in remove_chains:
                    pass
                if remove_water and res == 'HOH':
                    pass
                else:
                    lines.append(line[:22] + num + line[26:])

            elif line.startswith('TER') or line.startswith('END'):
                lines.append(line)

    if outfile is not None:
        with open(outfile, 'w') as f:
            for line in lines:
                f.writelines(line)

    return lines


def renumber_chain(pdb: str, chain: str, start_at: int = 1, outfile: str = None):
    """
    Renumbers a chain in a pdb file, starting from a defined start number.

    Parameters:
        pdb (str): path to file
        chain (str): chain id
        start_at (int, optional): start numbering from. Default 1
        outfile (str, optional): saves output file at target location.

    Returns:
        list: list of reordered pdb file
    """
    search = True
    const = 1
    lines = []
    with open(pdb, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == chain:
                if search:
                    search = False
                    if int(line[22:26]) < start_at:
                        const = start_at
                    else:
                        const = - start_at
                    num = format_integer(int(line[22:26]) + const)
                    lines.append(line[:22] + num + line[26:])
                elif line[21] == chain:
                    num = format_integer(int(line[22:26]) + const)
                    lines.append(line[:22] + num + line[26:])
            else:
                lines.append(line)

    if outfile is not None:
        with open(outfile, 'w') as f:
            for line in lines:
                f.writelines(line)

    return lines


def reorder_chains(pdb: str, order: list, outfile: str = None):
    """
    Reorderes chains in a pdb file in a specified order.

    Parameters:
        pdb (str): path to file
        order (list): desired order for chains
        outfile (str, optional): saves output file at target location

    Returns:
        list: lines of reordered pdb file.
    """
    chains = {}
    with open(pdb, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                if line[21] not in chains.keys():
                    chains[line[21]] = []
                else:
                    chains[line[21]].append(line)
    lines = []
    for key in order:
        for line in chains[key]:
            lines.append(line)
        lines.append('TER   \n')
    lines.append('END')

    if outfile is not None:
        with open(outfile, 'w') as f:
            for line in lines:
                f.writelines(line)

    return lines