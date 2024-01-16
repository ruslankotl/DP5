from rdkit import Chem
from rdkit.Chem import AllChem
from concurrent import futures
import argparse, logging, os, sys, time, copy, pickle

'''
algorithm borrowed from CASCADE paper and is used to run a quick estimate of conformation space
'''

dashedline = "   ------------------------------------------------------------------------------------------------------------------"
emptyline = "   |                                                                                                                     |"
normaltermination = "\n   -----------------       N   O   R   M   A   L      T   E   R   M   I   N   A   T   I   O   N      ----------------\n"
leftcol=97
rightcol=12

asciiArt = "     ___       ___                                    ___          ___          ___                   ___ \n    /  /\\     /__/\\                                  /__/\\        /  /\\        /__/\\         ___     /  /\\\n   /  /:/_    \\  \\:\\                                |  |::\\      /  /::\\       \\  \\:\\       /  /\\   /  /:/_\n  /  /:/ /\\    \\  \\:\\   ___     ___  ___     ___    |  |:|:\\    /  /:/\\:\\       \\  \\:\\     /  /:/  /  /:/ /\\\n /  /:/ /:/___  \\  \\:\\ /__/\\   /  /\\/__/\\   /  /\\ __|__|:|\\:\\  /  /:/  \\:\\  _____\\__\\:\\   /  /:/  /  /:/ /:/_\n/__/:/ /://__/\\  \\__\\:\\\\  \\:\\ /  /:/\\  \\:\\ /  /://__/::::| \\:\\/__/:/ \\__\\:\\/__/::::::::\\ /  /::\\ /__/:/ /:/ /\\\n\\  \\:\\/:/ \\  \\:\\ /  /:/ \\  \\:\\  /:/  \\  \\:\\  /:/ \\  \\:\\~~\\__\\/\\  \\:\\ /  /:/\\  \\:\\~~\\~~\\//__/:/\\:\\\\  \\:\\/:/ /:/\n \\  \\::/   \\  \\:\\  /:/   \\  \\:\\/:/    \\  \\:\\/:/   \\  \\:\\       \\  \\:\\  /:/  \\  \\:\\  ~~~ \\__\\/  \\:\\\\  \\::/ /:/\n  \\  \\:\\    \\  \\:\\/:/     \\  \\::/      \\  \\::/     \\  \\:\\       \\  \\:\\/:/    \\  \\:\\          \\  \\:\\\\  \\:\\/:/\n   \\  \\:\\    \\  \\::/       \\__\\/        \\__\\/       \\  \\:\\       \\  \\::/      \\  \\:\\          \\__\\/ \\  \\::/\n    \\__\\/     \\__\\/                                  \\__\\/        \\__\\/        \\__\\/                 \\__\\/\n  "

def SetupMM(settings):
    """
    Prepares Isomers for molecular mechanics calculation

    - settings: settings of the DP5 run

    Returns:
    ETKDGInputs: a list of inputs to be run by the conformational search program
    """
    ETKDGInputs = []

    for f in settings.InputFiles:

        if settings.Rot5Cycle is True:
            if not os.path.exists(f + 'rot.sdf'):
                import FiveConf
                # Generate the flipped fivemembered ring,
                # result is in '*rot.sdf' file
                FiveConf.main(f + '.sdf', settings)

        scriptdir = getScriptPath()
        cwd = os.getcwd()
        ETKDGInputs.append(f + '.sdf')

        if settings.Rot5Cycle is True:
            ETKDGInputs.append(f + 'rot.sdf')
        
        print(f'ETKDG input for {f} prepared.')

    return ETKDGInputs


def RunMM(ETKDGInputs, settings):
    """
    Runs the conformer search and saves its results.

    Returns
    ETKDGOutputs: a list of files containing completed results for calculations
    """
    efilter = settings.MaxCutoffEnergy/4.184
    ETKDGBaseNames = [x[:-4] for x in ETKDGInputs]
    ETKDGOutputs = []
    NCompleted = 0

    for isomer in ETKDGBaseNames:
        output_exists = os.path.exists(f'{isomer}_confs.sdf')
        if not output_exists:
            mol = Chem.MolFromMolFile(f'{isomer}.sdf', removeHs=False)
            confs, ids, nr = genConf(mol, nc=settings.HardConfLimit, rms=0.125,efilter=efilter,rmspost=0.5)
            save = Chem.SDWriter(f'{isomer}_confs.sdf')
            for energy, id in ids:
                conf = Chem.Mol(mol, confId=id)
                conf.SetDoubleProp('E', energy)
                conf.SetProp('_Name', '{}_{}'.format(isomer, id))
                save.write(conf)
            save.flush()
        ETKDGOutputs.append(f'{isomer}_confs.sdf')
    return ETKDGOutputs

def ReadConformers(ETKDGOutputs, Isomers, settings):
    for iso,outp in zip(Isomers,ETKDGOutputs):
        if (outp == f'{iso.BaseName}_confs.sdf'):
            print(outp + ' is matching conformational search output for ' + iso.BaseName)
            atoms, conformers, charge, AbsEs, mol = ReadETKDG(outp, settings)
            iso.Atoms = atoms
            iso.Conformers = conformers
            iso.MMCharge = charge
            iso.MMEnergies = AbsEs
            iso.Mol = mol
        else:
            continue
    return Isomers

def ReadETKDG(ETKDGOutput, settings):
    atoms = None
    conformers = []
    charge = 0
    energies = []
    suppl = Chem.SDMolSupplier(ETKDGOutput, removeHs=False)
    for conf in suppl:
        if atoms is None:
            mol = conf
            atoms = []
            for atom in conf.GetAtoms():
                atoms.append(atom.GetSymbol())
                charge+=(atom.GetFormalCharge())
        conf3d = conf.GetConformer()
        coords = conf3d.GetPositions().tolist()
        mol.AddConformer(conf3d,assignId=True)
        conformers.append(coords)
        energies.append(conf.GetDoubleProp('E'))
    return atoms, conformers, charge, energies, mol

def getScriptPath():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

# algorithm to generate nc conformations
def genConf(m, nc, rms, efilter, rmspost):
    """
    Generates the conformers.

    Parameters:
    - m(rdkit.Chem.Mol): the molecule on which a search is run
    - nc(int): number of conformers to sample
    - rms(float): RMSD threshold before MMFF optimisation
    - efilter: energy filter (kcal/mol)
    - rmspost: RMSD threshold after MMFF optimisation
    """

    nr = int(AllChem.CalcNumRotatableBonds(m))
    #m = Chem.AddHs(m)
    Chem.AssignAtomChiralTagsFromStructure(m, replaceExistingTags=True)
    if not nc: nc = min(1000, 3**nr)

    print(dashedline+"\n   |    "+("FULL_MONTE search").ljust(leftcol)+("|").rjust(rightcol))
    #log.Write("   | o  "+("COMP: "+str(Params.COMP)+" degrees").ljust(leftcol)+("|").rjust(rightcol))
    #log.Write("   | o  "+("LEVL: "+str(Params.LEVL)+" force field").ljust(leftcol)+("|").rjust(rightcol))
    #log.Write("   | o  "+("DEMX: "+str(Params.DEMX)+" kcal/mol").ljust(leftcol)+("|").rjust(rightcol))
    print("   | o  "+("EWIN: "+str(efilter)+" kcal/mol").ljust(leftcol)+("|").rjust(rightcol))
    print("   | o  "+("MCNV: "+str(nr)+" ROTATABLE BONDS").ljust(leftcol)+("|").rjust(rightcol))
    #log.Write("   |    "+torstring.ljust(leftcol)+("|").rjust(rightcol))
    print("   | o  "+("STEP: "+str(nc)+" (ESTIMATED CONFORMER SPACE: "+str(3**nr)+")").ljust(leftcol)+("|").rjust(rightcol))
    print(dashedline+"\n")

    if not rms: rms = -1
    ids=AllChem.EmbedMultipleConfs(m, numConfs=nc, randomSeed=0xf00d)


    if len(ids)== 0:
        ids = m.AddConformer(m.GetConformer, assignID=True)

    diz = []
    diz2 = []
    diz3 = []
    for id in ids:
        prop = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(m, prop, confId=id)
        ff.Minimize()
        en = float(ff.CalcEnergy())
        econf = (en, id)
        diz.append(econf)

    if efilter != "Y":
        n, diz2 = energy_filter(m, diz, efilter)
    else:
        n = m
        diz2 = diz

    if rmspost != None and n.GetNumConformers() > 1:
        o, diz3 = postrmsd(n, diz2, rmspost)
    else:
        o = n
        diz3 = diz2

    return o, diz3, nr

# filter conformers based on relative energy
def energy_filter(m, diz, efilter):
    print("o  FILTERING CONFORMERS BY ENERGY CUTOFF: "+str(efilter)+" kcal/mol")
    diz.sort()
    mini = float(diz[0][0])
    sup = mini + efilter
    n = Chem.Mol(m)
    n.RemoveAllConformers()
    n.AddConformer(m.GetConformer(int(diz[0][1])))
    nid = []
    ener = []
    nid.append(int(diz[0][1]))
    ener.append(float(diz[0][0])-mini)
    del diz[0]
    for x,y in diz:
        if x <= sup:
            #print("   KEEP - Erel:", x-mini)
            n.AddConformer(m.GetConformer(int(y)))
            nid.append(int(y))
            ener.append(float(x-mini))
        else:
            #print("   REMOVE - Erel:", x-mini)
            break
    diz2 = list(zip(ener, nid))
    print("   KEEPING "+str(len(ener))+" CONFORMERS")
    return n, diz2

# filter conformers based on geometric RMS
def postrmsd(n, diz2, rmspost):
    print("o  FILTERING CONFORMERS BY RMS: "+str(rmspost))
    diz2.sort(key=lambda x: x[0])
    o = Chem.Mol(n)
    confidlist = [diz2[0][1]]
    enval = [diz2[0][0]]
    nh = Chem.RemoveHs(n)
    del diz2[0]
    for z,w in diz2:
        confid = int(w)
        p=0
        for conf2id in confidlist:
            #print(confid, conf2id)
            rmsd = AllChem.GetBestRMS(nh, nh, prbId=confid, refId=conf2id)
            if rmsd < rmspost:
                p=p+1
                break
        if p == 0:
            confidlist.append(int(confid))
            enval.append(float(z))
    diz3 = list(zip(enval, confidlist))
    print("   KEEPING "+str(len(enval))+" CONFORMERS")
    return o, diz3