import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from dp5.conformer_search.base_cs_method import BaseConfSearch, ConfData
'''
algorithm borrowed from CASCADE paper and is used to run a quick estimate of conformation space
'''
logger = logging.getLogger(__name__)

class ConfSearchMethod(BaseConfSearch):

    def __init__(self, inputs, settings):
        super().__init__(inputs,settings)


    def prepare_input(self):
        # no conversion required
        return self.inputs
    
    def __repr__(self) -> str:
        return "ETKDG"

    def run(self):
        logger.info(f"Using {self} as conformer search method")
        self.inputs = self.prepare_input()
        self.outputs = self._run()
        logger.debug(f"Conformer search output: {self.outputs}")
        
        return self.parse_output()

    def _run(self):
        efilter = self.settings['energy_cutoff'] / 4.184
        conf_limit = self.settings['conf_limit']

        outputs = []
        for input in self.inputs:
            output_exists = Path(f'{input}.confs').exists()
            if not output_exists:
                mol = Chem.MolFromMolFile(f'{input}.sdf', removeHs=False)
                confs, ids, nr = self._genConf(mol, nc=conf_limit, rms=0.125,efilter=efilter, rmspost=0.5)
                save = Chem.SDWriter(f'{input}.confs')
                for energy, id in ids:
                    conf = Chem.Mol(mol, confId=id)
                    conf.SetProp('E', f'{energy * 4.184:.2f}')
                    conf.SetProp('_Name', '{}_{}'.format(input, id))
                    save.write(conf)
                save.flush()
            outputs.append(f'{input}.confs')
        return outputs

    def _parse_output(self, file):
        out_file = f"{file}.confs"
        atoms = None
        conformers = []
        charge = 0
        energies = []
        suppl = Chem.SDMolSupplier(out_file, removeHs=False)
        for conf in suppl:
            if atoms is None:
                atoms = []
                for atom in conf.GetAtoms():
                    atoms.append(atom.GetSymbol())
                    charge+=(atom.GetFormalCharge())
            conf3d = conf.GetConformer()
            coords = conf3d.GetPositions().tolist()
            conformers.append(coords)
            energies.append(float(conf.GetProp('E')))
            conf_data = ConfData(atoms,conformers,charge, energies)
        return conf_data 
                
    # algorithm to generate nc conformations
    def _genConf(self, m, nc, rms, efilter, rmspost):
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
        Chem.AssignAtomChiralTagsFromStructure(m, replaceExistingTags=True)
        if not nc: nc = min(1000, 3**nr)

        logger.info("ETKDG conformational search")
        logger.info(f"Energy window: {efilter:.2f} kcal/mol")
        logger.debug(f"Will generate up to {nc} conformers")

        if not rms: rms = -1
        ids=AllChem.EmbedMultipleConfs(m, numConfs=nc, randomSeed=0xf00d, useRandomCoords=True, pruneRmsThresh=rms)


        if len(ids)== 0:
            ids = m.AddConformer(m.GetConformer, assignID=True)

        logger.info(f"Generated {len(ids)} conformers")
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
            n, diz2 = self._energy_filter(m, diz, efilter)
        else:
            n = m
            diz2 = diz

        if rmspost != None and n.GetNumConformers() > 1:
            o, diz3 = self._postrmsd(n, diz2, rmspost)
        else:
            o = n
            diz3 = diz2

        return o, diz3, nr

    # filter conformers based on relative energy
    def _energy_filter(self, m, diz, efilter):
        logger.info("Filtering conformers, energy cutoff: %.2f kcal/mol", efilter)
        diz.sort()
        mini = float(diz[0][0])
        sup = mini + efilter
        n = Chem.Mol(m)
        n.RemoveAllConformers()
        n.AddConformer(m.GetConformer(int(diz[0][1])))
        nid = []
        ener = []
        nid.append(int(diz[0][1]))
        ener.append(float(diz[0][0]))
        del diz[0]
        for x,y in diz:
            if x <= sup:
                n.AddConformer(m.GetConformer(int(y)))
                nid.append(int(y))
                ener.append(float(x))
            else:
                break
        diz2 = list(zip(ener, nid))
        logger.info(f"Retained {len(ener)} conformers")
        return n, diz2

    # filter conformers based on geometric RMS
    def _postrmsd(self, n, diz2, rmspost):
        logger.info(f"Filtering conformers, RMS cutoff: {rmspost}")
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
        logger.info("Retained %s conformers", len(enval))
        return o, diz3
    