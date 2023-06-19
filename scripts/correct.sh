#!/bin/bash

for file in hallucinations/*.pdb; do
    # count the number of lines in the file
    num_lines=$(wc -l < $file)
    
    # remove the last 4 lines
    head -n $(($num_lines-4)) $file > temp.pdb
    
    # add TER and END to the end of the file
    echo "TER" >> temp.pdb
    echo "END" >> temp.pdb
    
    # replace the original file with the modified file
    mv temp.pdb $file
done