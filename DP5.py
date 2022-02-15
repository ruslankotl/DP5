import numpy as np
import pickle
from scipy.stats import gmean
from pathlib import Path
from scipy import stats
import os
import pathos.multiprocessing as mp
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem
try:
    from openbabel.openbabel import OBConversion, OBMol, OBAtomAtomIter, OBMolAtomIter
except ImportError:
    from openbabel import *

import pandas as pd
import CNN_model

c_distance = 4.532297920317418

class DP5data:

    def __init__(self,ScriptPath,Atoms):

        self.Atom_number = Atoms

        self.Cshifts = []  # Carbon shifts used in DP5 calculation
        self.Cscaled = []  # Scaled Carbon shifts used in DP5 calculation
        self.Cexp = []  # Carbon experimental shifts used in DP4 calculation
        self.Clabels = []  # Carbon atom labels
        self.Cinds = [] # Carbon atom indcies
        self.Hshifts = []  # Proton shifts used in DP5 calculation
        self.Hscaled = []  # Scaled Proton shifts used in DP5 calculation
        self.Hexp = []  # Proton experimental shifts used in DP5 calculation
        self.Hlabels = []  # Proton atom labels
        self.CMAE = [] # Carbon Mean Absolute error
        self.CMax = [] # Carbon Max Absolute error
        self.ConfCshifts = []

        self.Mols = [] #qml compound objects for each isomer

        self.ErrorAtomReps = [] # FCHL representations ordered by Clabels

        self.ErrorAtomProbs = [] #per atom dp5 scaled probabilities for all conformers

        self.B_ErrorAtomProbs = [] #per atom dp5 Exp probabilities boltzmann weighted

        self.Mol_Error_probs = []  # DP5 for isomers based on Carbon data

        self.DP5_Error_probs = []  # Final DP5S

        self.ExpAtomReps = [] # FCHL representations ordered by Clabels

        self.ExpAtomProbs = [] #per atom dp5 scaled probabilities for all conformers

        self.B_ExpAtomProbs = [] #per atom dp5 Exp probabilities boltzmann weighted

        self.Mol_Exp_probs = []  # DP5 for isomers based on Carbon data

        self.DP5_Exp_probs = []  # Final DP5S

        self.output = ""


def ProcessIsomers(dp5Data, Isomers, Settings):

    # extract calculated and experimental shifts and add to dp5Data instance

    # Carbon

    # make sure any shifts with missing peaks are removed from all isomers

    for iso in Isomers:

        dp5Data.Cexp.append([])
        dp5Data.Cshifts.append([])
        dp5Data.Clabels.append([])
        dp5Data.Cinds.append([])
        dp5Data.ConfCshifts.append([])

        a_ind = 0

        exp_inds = []

        for shift, exp, label in zip(iso.Cshifts, iso.Cexp, iso.Clabels):

            if exp != '':

                dp5Data.Cshifts[-1].append(shift)
                dp5Data.Cexp[-1].append(exp)
                dp5Data.Clabels[-1].append(label)
                dp5Data.Cinds[-1].append(int(label[1:]) - 1)
                exp_inds.append(a_ind)

            a_ind += 1

        if len(iso.ConformerCShifts) > 0:

            for conf_shifts in iso.ConformerCShifts:

                dp5Data.ConfCshifts[-1].append( [conf_shifts[e] for e in exp_inds])

    # if the is nmr data append scaled shifts

    if "n" in Settings.Workflow:

        for iso in range(0, len(Isomers)):

            if len(dp5Data.Cshifts[iso]) > 3:

                dp5Data.Cscaled.append(ScaleNMR(dp5Data.Cshifts[iso], dp5Data.Cexp[iso]))

            else:

                dp5Data.Cscaled.append(dp5Data.Cshifts[iso])

            dp5Data.CMAE.append(np.mean(np.abs(np.array(dp5Data.Cscaled[iso]) - np.array(dp5Data.Cexp[iso]))))
            dp5Data.CMax.append(np.max(np.abs(np.array(dp5Data.Cscaled[iso]) - np.array(dp5Data.Cexp[iso]))))

    for iso_ind , iso in enumerate(Isomers):

        InputFile = Path(iso.InputFile)

        # make rdkit mol for each conformer

        dp5Data.Mols.append([])

        if ("m" not in Settings.Workflow) & ("o" not in Settings.Workflow):

            dp5Data.Mols[-1].append(iso.Mols[0])

        else:

            conf_data = []

            if "o" in Settings.Workflow:

                conf_data = iso.DFTConformers

            elif "m" in Settings.Workflow:

                conf_data = iso.Conformers

            for i, geom in enumerate(conf_data):

                m = Chem.MolFromMolFile(str(InputFile) + ".sdf", removeHs=False)

                conf = m.GetConformer(0)

                for j, atom_coords in enumerate(geom):
                    conf.SetAtomPosition(j, Point3D(float(atom_coords[0]), float(atom_coords[1]),
                                                    float(atom_coords[2])))

                dp5Data.Mols[-1].append(m)


    # make pandas df to go into CNN predicting model

    if "n" in Settings.Workflow:

        model = CNN_model.build_model(Settings, "Error")


    else:

        model = CNN_model.build_model(Settings, "Exp")


    for iso in range(len(Isomers)):

        iso_df = []

        i = 0
        for conf in dp5Data.Mols[iso]:
            iso_df += [(i, conf, np.array(dp5Data.Cinds[iso]))]

            i += 1

        iso_df = pd.DataFrame(iso_df, columns=['conf_id', 'Mol', 'atom_index'])

        # use CNN to get reps for each conformer

        if "n" in Settings.Workflow:

            dp5Data.ErrorAtomReps.append(CNN_model.extract_Error_reps(model, iso_df, Settings))

        else:

            dp5Data.ExpAtomReps.append(CNN_model.extract_Exp_reps(model, iso_df, Settings))

    return dp5Data


def kde_probs(Isomers,Settings,DP5type, AtomReps, ConfCshifts,Cexp):

    if DP5type == "Error":

        Error_kernel = pickle.load(open(Path(Settings.ScriptDir) / "pca_10_kde_ERRORrep_Error_kernel.p", "rb"))

        def kde_probfunction(conf_shifts, conf_reps, exp_data):

            # loop through atoms in the test molecule - generate kde for all of them.

            # check if this has been calculated

            n_points = 250

            x = np.linspace(-20, 20, n_points)

            ones_ = np.ones(n_points)

            n_comp = 10

            p_s = []

            scaled_shifts = ScaleNMR(conf_shifts, exp_data)

            scaled_errors = [shift - exp for shift, exp in zip(scaled_shifts, exp_data)]

            for rep, value in zip(conf_reps, scaled_errors):

                # do kde hybrid part here to yield the atomic probability p

                point = np.vstack((rep.reshape(n_comp, 1) * ones_, x))

                pdf = Error_kernel.pdf(point)

                integral_ = np.sum(pdf)

                if integral_ != 0:
                    max_x = x[np.argmax(pdf)]

                    low_point = max(-20, max_x - abs(max_x - value))

                    high_point = min(20, max_x + abs(max_x - value))

                    low_bound = np.argmin(np.abs(x - low_point))

                    high_bound = np.argmin(np.abs(x - high_point))

                    bound_integral = np.sum(pdf[min(low_bound, high_bound): max(low_bound, high_bound)])

                    p_s.append(bound_integral / integral_)

            return p_s

    if DP5type == "Exp":

        Exp_kernel = pickle.load( open(Path(Settings.ScriptDir) / "pca_10_kde_EXP_kernel.p", "rb"))

        def kde_probfunction(_, conf_reps, exp_data):

            # loop through atoms in the test molecule - generate kde for all of them.

            # check if this has been calculated

            n_points = 250

            x = np.linspace(0, 250, n_points)

            ones_ = np.ones(n_points)

            n_comp = 10

            p_s = []

            for rep, value in zip(conf_reps ,exp_data):

                # do kde hybrid part here to yield the atomic probability p

                point = np.vstack((rep.reshape(n_comp, 1) * ones_, x))

                pdf = Exp_kernel.pdf(point)

                integral_ = np.sum(pdf)

                if integral_ != 0:

                    max_x = x[np.argmax(pdf)]

                    low_point = max(0, max_x - abs(max_x - value))

                    high_point = min(250, max_x + abs(max_x - value))

                    low_bound = np.argmin(np.abs(x - low_point))

                    high_bound = np.argmin(np.abs(x - high_point))

                    bound_integral = np.sum(pdf[min(low_bound, high_bound): max(low_bound, high_bound)])

                    p_s.append(bound_integral / integral_)

                else:

                    p_s.append(1)

            return p_s

    #for each atom in the molecule calculate the atomic worry factor

    AtomProbs = [[] for i in range(len(Isomers))]

    for iso in range(len(Isomers)):

        res = [[] for i in AtomReps[iso]]

        AtomProbs[iso] = [[] for i in AtomReps[iso]]

        maxproc = 4

        pool = mp.Pool(maxproc)

        if len(ConfCshifts) > 0:

            conf_shifts = ConfCshifts[iso]

        else:

            conf_shifts = [[] for i in AtomReps[iso] ]

        ind1 = 0

        for shifts , conf_reps in zip(conf_shifts , AtomReps[iso])  :


            res[ind1] = pool.apply_async(kde_probfunction,
                                         [conf_shifts[ind1],conf_reps,Cexp[iso]])

            ind1 += 1

        for ind1 in range(len(res)):

            AtomProbs[iso][ind1] = res[ind1].get()

    return AtomProbs


def ScaleNMR(calcShifts, expShifts):

    slope, intercept, r_value, p_value, std_err = stats.linregress(expShifts,
                                                                   calcShifts)
    scaled = [(x - intercept) / slope for x in calcShifts]

    return scaled


def BoltzmannWeight_DP5(Isomers,AtomProbs):

    BAtomProbs = []

    for iso,scaled_probs in zip( Isomers, AtomProbs):

        B_scaled_probs = [0] * len(scaled_probs[0])

        for population, conf_scaled_p in zip(iso.Populations, scaled_probs ):

            for i in range(len(B_scaled_probs)):

                B_scaled_probs[i] += conf_scaled_p[i] * population

        BAtomProbs.append([1 - p for p in B_scaled_probs])

    return BAtomProbs


def Calculate_DP5(BAtomProbs):

    Molecular_probability = []

    for scaled_probs in BAtomProbs:

        #Molecular_probability.append( 1.0 - gmean(scaled_probs))

        Molecular_probability.append(gmean([ p_si for p_si in scaled_probs]))

        #Molecular_probability.append(np.product([1 - p_si for p_si in scaled_probs]))

        #Molecular_probability.append( 1 - np.product([ p_si for p_si in scaled_probs]))

    return Molecular_probability

def Exp_scaling_function(x):

    # Empirical Scaling Function

    return 0.72 / (1 + np.exp(- 17.16382728 * (x - 0.28914008)))


def Rescale_DP5(Mol_probs,BAtomProbs,Settings,DP5type,CMAE):

    if DP5type == "Exp":

        incorrect_kde = pickle.load(open(Path(Settings.ScriptDir) / "Exp_incorrect_kde.p" ,"rb"))

        correct_kde = pickle.load(open(Path(Settings.ScriptDir) / "Exp_correct_kde.p" ,"rb"))

        DP5AtomProbs = [ [] for iso in range(0,len(Mol_probs)) ]

        DP5probs = []

        for iso in range(0, len(Mol_probs)):

            DP5AtomProbs[iso] = [ Exp_scaling_function(x) for x in BAtomProbs[iso]]

            DP5probs.append(float( Exp_scaling_function(Mol_probs[iso]) ))

            #DP5AtomProbs[iso] = [ x for x in BAtomProbs[iso]]

            #DP5probs.append(float( Mol_probs[iso]) )

            #DP5AtomProbs[iso] = [float(correct_kde.pdf(x) / (incorrect_kde.pdf(x) + correct_kde.pdf(x))) for x in
                                # BAtomProbs[iso]]

            #DP5probs.append(float(correct_kde.pdf(Mol_probs[iso]) / (
                        #incorrect_kde.pdf(Mol_probs[iso]) + correct_kde.pdf(Mol_probs[iso]))))



    elif DP5type == "Error":

        incorrect_kde = pickle.load(open(Path(Settings.ScriptDir) / "Error_incorrect_kde.p" ,"rb"))

        correct_kde = pickle.load(open(Path(Settings.ScriptDir) / "Error_correct_kde.p" ,"rb"))

        DP5AtomProbs = [ [] for iso in range(0,len(Mol_probs)) ]

        DP5probs = []

        print("CMAE" , CMAE)

        for iso in range(0, len(Mol_probs)):

            if CMAE[iso] < 2:

                DP5AtomProbs[iso] = [float(correct_kde.pdf(x) / (incorrect_kde.pdf(x) + correct_kde.pdf(x))) for x in BAtomProbs[iso]]

                DP5probs.append(float(correct_kde.pdf(Mol_probs[iso]) / (incorrect_kde.pdf(Mol_probs[iso]) + correct_kde.pdf(Mol_probs[iso]))))


            else:

                DP5AtomProbs[iso] = BAtomProbs[iso]

                DP5probs.append(Mol_probs[iso])

    else:

        DP5probs = []

        DP5AtomProbs = []

    return DP5probs, DP5AtomProbs


def Pickle_res(dp5Data,Settings):

    data_dic = {"Cshifts": dp5Data.Cshifts,
                "Cscaled": dp5Data.Cscaled,
                "Cexp": dp5Data.Cexp,
                "CMAE": dp5Data.CMAE,
                "CMax": dp5Data.CMax,
                "Clabels": dp5Data.Clabels,
                "Hshifts": dp5Data.Hshifts,
                "Hscaled": dp5Data.Hscaled,
                "Hexp": dp5Data.Hexp,
                "Hlabels": dp5Data.Hlabels,
                "ConfCshifts": dp5Data.ConfCshifts,

                "Mols": dp5Data.Mols,

                "ErrorAtomReps" : dp5Data.ErrorAtomReps,

                "ErrorAtomProbs" : dp5Data.ErrorAtomProbs,

                "B_ErrorAtomProbs" : dp5Data.B_ErrorAtomProbs ,

                "Mol_Error_probs" : dp5Data.Mol_Error_probs ,

                "DP5_Error_probs" : dp5Data.DP5_Error_probs ,

                "ExpAtomReps" : dp5Data.ExpAtomReps ,

                "ExpAtomProbs" : dp5Data.ExpAtomProbs ,

                "B_ExpAtomProbs" : dp5Data.B_ExpAtomProbs ,

                "Mol_Exp_probs" : dp5Data.Mol_Exp_probs ,

                "DP5_Exp_probs" : dp5Data.DP5_Exp_probs

                }

    pickle.dump(data_dic , open(Path(Settings.OutputFolder) / "dp5" / "data_dic.p","wb"))

    return dp5Data


def UnPickle_res(dp5Data,Settings):

    data_dic =  pickle.load(open(Path(Settings.OutputFolder) / "dp5" / "data_dic.p","rb"))

    dp5Data.Cshifts = data_dic["Cshifts"]
    dp5Data.Cscaled = data_dic["Cscaled"]
    dp5Data.Cexp = data_dic["Cexp"]
    dp5Data.CMAE = data_dic["CMAE"]
    dp5Data.CMax = data_dic["CMax"]
    dp5Data.Clabels = data_dic["Clabels"]
    dp5Data.Hshifts = data_dic["Hshifts"]
    dp5Data.Hscaled = data_dic["Hscaled"]
    dp5Data.Hexp = data_dic["Hexp"]
    dp5Data.Hlabels = data_dic["Hlabels"]
    dp5Data.ConfCshifts = data_dic["ConfCshifts"]

    dp5Data.Mols = data_dic["Mols"]
    dp5Data.ErrorAtomReps= data_dic["ErrorAtomReps"]
    dp5Data.ErrorAtomProbs= data_dic["ErrorAtomProbs"]
    dp5Data.B_ErrorAtomProbs= data_dic["B_ErrorAtomProbs"]
    dp5Data.Mol_Error_probs= data_dic["Mol_Error_probs"]
    dp5Data.DP5_Error_probs= data_dic["DP5_Error_probs"]
    dp5Data.ExpAtomReps= data_dic["ExpAtomReps"]
    dp5Data.ExpAtomProbs= data_dic["ExpAtomProbs"]
    dp5Data.B_ExpAtomProbs= data_dic["B_ExpAtomProbs"]
    dp5Data.Mol_Exp_probs= data_dic["Mol_Exp_probs"]
    dp5Data.DP5_Exp_probs = data_dic["DP5_Exp_probs"]

    return dp5Data


def PrintAssignment(dp5Data,DP5AtomProbs,output,Settings):

    isomer = 0

    if "n" in Settings.Workflow:

        for Clabels, Cshifts, Cexp, Cscaled, atom_p in zip(dp5Data.Clabels, dp5Data.Cshifts, dp5Data.Cexp, dp5Data.Cscaled,DP5AtomProbs):

            dp5Data.output += ("\n\nAssigned C shifts for isomer " + str(isomer + 1) + ": ")

            output = PrintNMR(Clabels, Cshifts, Cscaled, Cexp,atom_p, output)

            isomer += 1

    else:

        for Clabels, Cexp,  atom_p in zip(dp5Data.Clabels, dp5Data.Cexp, DP5AtomProbs):

            dp5Data.output += ("\n\nAssigned C shifts for isomer " + str(isomer + 1) + ": ")

            output = PrintNMR_EXP(Clabels, Cexp, atom_p, output)

            isomer += 1

    return output


def PrintNMR(labels, values, scaled, exp,atom_p,output):

    s = np.argsort(values)

    svalues = np.array(values)[s]

    slabels = np.array(labels)[s]

    sscaled = np.array(scaled)[s]

    sexp = np.array(exp)[s]

    atom_p = np.array(atom_p)[s]

    output += ("\nlabel, calc, corrected, exp, error,prob")

    for i in range(len(labels)):

        output += ("\n" + format(slabels[i], "6s") + ' ' + format(svalues[i], "6.2f") + ' '
                           + format(sscaled[i], "6.2f") + ' ' + format(sexp[i], "6.2f") + ' ' +
                           format(sexp[i] - sscaled[i], "6.2f")+ ' ' +
                           format(atom_p[i] , "6.2f"))

    return output


def PrintNMR_EXP(labels, values, atom_p,output):

    s = np.argsort(values)

    svalues = np.array(values)[s]

    slabels = np.array(labels)[s]

    atom_p = np.array(atom_p)[s]

    output += ("\nlabel, exp, error,prob")

    for i in range(len(labels)):

        output += ("\n" + format(slabels[i], "6s") + ' ' + format(svalues[i], "6.2f") + ' '

                            + format(1 - atom_p[i] , "6.2f"))

    return output


def MakeOutput( Isomers, Settings,DP5data,DP5Probs,DP5AtomProbs):

    # add some info about the calculation

    output = ""

    output += Settings.InputFiles[0] + "\n"

    output += "\n" + "Solvent = " + Settings.Solvent

    output += "\n" + "Force Field = " + Settings.ForceField + "\n"

    if 'o' in Settings.Workflow:
        output += "\n" + "DFT optimisation Functional = " + Settings.oFunctional
        output += "\n" + "DFT optimisation Basis = " + Settings.oBasisSet

    if 'e' in Settings.Workflow:
        output += "\n" + "DFT energy Functional = " + Settings.eFunctional
        output += "\n" + "DFT energy Basis = " + Settings.eBasisSet

    if 'n' in Settings.Workflow:
        output += "\n" + "DFT NMR Functional = " + Settings.nFunctional
        output += "\n" + "DFT NMR Basis = " + Settings.nBasisSet

    if Settings.StatsParamFile != "none":
        output += "\n\nStats model = " + Settings.StatsParamFile

    output += "\n\nNumber of isomers = " + str(len(Isomers))

    c = 1

    for i in Isomers:
        output += "\nNumber of conformers for isomer " + str(c) + " = " + str(len(i.Conformers))

        c += 1

    output = PrintAssignment(DP5data,DP5AtomProbs,output,Settings)

    output += ("\n\nResults of DP5: ")

    for i, p in enumerate(DP5Probs):

        output += ("\nIsomer " + str(i + 1) + ": " + format(p * 100, "4.1f") + "%")

    print(output)

    if Settings.OutputFolder == '':

        out = open(str(os.getcwd()) + "/" + str("Output.dp5"), "w+")

    else:

        out = open(os.path.join(Settings.OutputFolder, str("Output.dp5")), "w+")

    out.write(output)

    out.close()

    return output



def predict_Exp_shifts(Settings, Isomers):

    Exp_model = CNN_model.build_Exp_predicting_model(Settings)

    iso_df = []

    i =0

    labels = []

    for iso in Isomers:

        labels.append([])

        InputFile = Path(iso.InputFile)

        # make rdkit mol for each conformer

        m = Chem.MolFromMolFile(str(InputFile) + ".sdf", removeHs=False)

        #else use rdkit optimisation

        AllChem.EmbedMolecule(m, useRandomCoords=True)

        AllChem.MMFFOptimizeMolecule(m)

        Isomers[i].Mols.append(m)

        Isomers[i].Populations.append(1)

        inds = []

        #else use starting point geom.

        for  j , atom in enumerate(m.GetAtoms()):

            if atom.GetAtomicNum() == 6:
                inds.append(j)

                labels[-1].append("C" + str(j + 1))

        iso_df.append((i , m , np.array(inds) ))

        i+=1


    iso_df = pd.DataFrame( iso_df, columns=['conf_id', 'Mol', 'atom_index'])

    shifts = CNN_model.predict_shifts(Exp_model, iso_df ,Settings)

    return shifts,labels,Isomers