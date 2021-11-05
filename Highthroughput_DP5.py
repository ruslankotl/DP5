import os
import argparse
from from pathlib import Path
import shutil
import re

parser = argparse.ArgumentParser()

# The input file is a text file arranged in two columns
# All The required NMR data must be in the cwd when this script is launched
# the first line contains the structure input type e.g InChI SMILES or SMARTS
# the following lines are separated into two columns
# the first contains the input strings enclosed in square brackets and separated by commas if multiple strcutres are proposed for the same NMR data
# the second column contains the NMR data file/folder name  - this file/ folder must be in the cwd when this script is called

# e.g

# InChI
# [InChI/SMILES/SMARTS mol 1 ]                                                                       NMR_data_file Name mol 1
# [InChI/SMILES/SMARTS mol 2a , InChI/SMILES/SMARTS mol 2b ... InChI/SMILES/SMARTS mol 2z]           NMR_data_file Name mol 2
#                               .
#                               .
#                               .
# [InChI/SMILES/SMARTS mol n ]                                                                       NMR_data_file Name mol n


# The output folder must be specified
# the input data will be copied to this location and DP5 called from there.


parser.add_argument('--InputFile', type=list)

parser.add_argument('--OutputFolder', type=str)

parser.add_argument('--PyDP4Settings' , type=str)

args = parser.parse_args()

InputFile = args.InputFile

OutputFolder = args.OutputFolder

PyDP4Settings = args.PyDP4Settings

f = open(InputFile,"r")

line_n = 0

input_type = ""

for line in f.readlines():

    if line_n ==0:

        input_type = line.strip()


    else:

        inps = line.split()[0]

        NMR_location = line.split()[1]

        #now make a new file in the output location to copy the inputs to

        os.mkdir( Path(OutputFolder) / "Mol_" + str(line_n))

        # copy the NMR input files to this location

        shutil.copytree( Path.cwd() / NMR_location , Path(OutputFolder) / "Mol_" + str(line_n) )

        # make a newfile with the input strings in the new file location

        os.chdir(Path(OutputFolder) / "Mol_" + str(line_n))

        inp_file = open("Mol_" + str(line_n) + "." + input_type.lower )

        inps = re.sub(r"]|\[|\s", "", inps)

        inps = inps.split(",")

        for i in inps[:-1]:

            inp_file.write(i + "\n")

        inp_file.write(inps[-1])

    line_n +=1

#now run the PyDP4commands