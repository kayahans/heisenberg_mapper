#!/usr/bin/env python
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer
from pymatgen.io.vasp.outputs import Vasprun, Outcar
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CutOffDictNN
import glob, pdb
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


directories = glob.glob("./*/")

def parse_dirs(structures, energies, species, magmoms, magmoms_all):
    try:
        # Try to load first if parsed before
        structures = pickle.load(open("structures.p", "rb"))
        energies   = pickle.load(open("energies.p",   "rb" ))
    
    except:
        for dirs in directories:
            energy = 0
            try:
                vasprun     = Vasprun(dirs+'/vasprun.xml')
                structure   = vasprun.ionic_steps[-1]['structure']
                energy      = vasprun.ionic_steps[-1]['e_wo_entrp']
            except:
                try:
                    structure = Structure.from_file(dirs+'/CONTCAR')
                except:
                    print("vasprun.xml or CONTCAR at {} could not be read successfully! Exiting".format(dirs))
                    exit()
                #end 
            #end
            try:
                outcar      = Outcar(dirs+"/OUTCAR")
                magnetization     = outcar.magnetization
                if energy == 0:
                    energy = outcar.final_energy
            except:
                print("OUTCAR at {} could not be read successfully! Exiting".format(dirs))
                exit()
            #end 
            
            magmom      = [x['tot'] for x in magnetization]
            structure.add_site_property("magmom", magmom)
            mag_str     = CollinearMagneticStructureAnalyzer(structure, threshold_nonmag = 0.5, make_primitive=False).get_structure_with_only_magnetic_atoms(make_primitive=False)
            magmom      = mag_str.site_properties['magmom']

            structures.append(mag_str)
            energies.append(energy)
            species.append([ii.name for ii in mag_str.species])
            magmoms.append(magmom)
            magmoms_all.append(CollinearMagneticStructureAnalyzer(structure, make_primitive=False).magmoms)

            # Print report
            print("Number of directories parsed: {}".format(len(directories)))
            print("Name of directories: ", directories)
            print("Saving to structures.p and energies.p")
            if len(structures) != 0 and len(energies) != 0:
                pickle.dump(structures, open( "structures.p", "wb" ) )
                pickle.dump(energies,   open( "energies.p", "wb" ) )
            print("Saved to .p files")
    #end try

    print("Printing input")
    print("Lattice(A) Energy(eV) Magmoms(m_B) Atoms")
    for idx in range(len(structures)):
        structure = structures[idx]
        energy = energies[idx]
        magmoms = structure.site_properties['magmom']
        atoms = [x.name for x in structure.species]
        print("{} {} {} {}".format(structure.lattice.abc, energy, magmoms, atoms))
    #end for
    print("End of printing input\n")
    return structures, energies, species, magmoms, magmoms_all
#end def parse_dirs

def get_zero_dict_with_keys(input_dict):
    "Works with only two layers of keys"
    new_dict = dict.fromkeys(input_dict)
    for key in new_dict.keys():
        new_dict[key] = dict.fromkeys(input_dict[key], 0)
    return new_dict


def get_key_tree(input_dict):
    "Works with only two layers of keys"
    tree_list = []
    for key1 in input_dict.keys():
        for key2 in input_dict[key1].keys():
            tree_list.append(key1+'-'+key2)
    return tree_list

class HeisenbergMapper:
    def __init__(self, structures, energies):
        self.structures = structures
        self.energies = energies
        natoms_max = -1
        for structure in structures:
            if structure.num_sites > natoms_max:
                natoms_max = structure.num_sites

        self.natoms_max = natoms_max

        if len(self.structures) < 2:
            print("With less than 2 structures, it is not possible to make a fit. Num structures : {}. Exiting!".format(len(structures)))
            exit()
        #end if

    def get_ex_mat(self, interactions, cutoff_nn):
        columns = ["E", "E0"]
        interaction_list = get_key_tree(interactions)
        self.num_interactions = len(interaction_list)
        for key in interaction_list:
            columns.append(key)
        ex_mat = pd.DataFrame(columns=columns)
        
        sgraph_index = 0
        for str_idx, structure in enumerate(self.structures):
            spins = get_zero_dict_with_keys(interactions)
            counts = get_zero_dict_with_keys(interactions)
            magmom = np.round(structure.site_properties['magmom'])
            for site_idx, site in enumerate(structure):
                neighbors = cutoff_nn.get_nn_info(structure, site_idx)
                for nn in neighbors:
                    site_name = site.specie.name
                    neigh_name = nn['site'].specie.name
                    neigh_idx = nn['site_index']
                    bond_length = nn['weight']
                    bond = '-'.join(sorted([neigh_name, site_name]))
                    if bond in interactions.keys():
                        for bond_pair in interactions[bond].keys():
                            if interactions[bond][bond_pair][0] < bond_length < interactions[bond][bond_pair][1]:
                                spins[bond][bond_pair] += 1./2 * \
                                    (magmom[site_idx]) * (magmom[neigh_idx])

            ex_row = pd.DataFrame(np.zeros((1, self.num_interactions+2)), index=[
                                  sgraph_index], columns=columns)
            scale = self.natoms_max/structure.num_sites
            ex_row.at[sgraph_index, "E"] = self.energies[str_idx] * scale
            ex_row.at[sgraph_index, "E0"] = 1
            for atom_pair, pair_dict in spins.items():
                for neigh, s_ij in pair_dict.items():
                    key = atom_pair + '-' + neigh
                    ex_row.at[sgraph_index, key] = s_ij * scale
            ex_mat = ex_mat.append(ex_row)
            sgraph_index += 1
        # end loop over structures
        self.ex_mat = ex_mat
        self.E = ex_mat[["E"]]
        H = ex_mat.loc[:, ex_mat.columns != "E"].values
        self.H = H

    def fit(self):
        # Fit all the data using least squares
        self.j_ij = np.linalg.lstsq(self.H, self.E, rcond=None)[0]
        self.E_fit = np.dot(self.H, self.j_ij)
        self.mad = np.average(np.abs(self.E-self.E_fit))
        self.j_ij[1:] *= 1000
        j_names = [j for j in self.ex_mat.columns if j not in ["E"]]
        self.ex_params = {j_name: j[0]
                           for j_name, j in zip(j_names, self.j_ij)}
        
    def plot(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.E, self.E_fit)
        y = self.E.copy()
        y = np.array(sorted(y.values))
        plt.plot(y, y, '-k')
        plt.xlabel("DFT energy (eV)")
        plt.ylabel("Fitted energy (eV)")
        plt.show()
    
    def report(self):
        print("MAD = {} num_param = {}".format(np.round(self.mad,4), self.num_interactions))
        print("Units: E0=eV, nnx=meV")
        print(self.ex_params)
        print(self.ex_mat)
    
    def cross_validation(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import LeaveOneOut
        from itertools import combinations

        X = np.array(self.H)
        y = np.array(self.E)
        
        for num_params in range(1, self.num_interactions+2):
            print("Number of parameters = {}".format(num_params))
            test_error_min = 10e6
            for x_i in combinations(enumerate(X.T), r=num_params):
                param_list = []
                x_i_new = []
                for i in range(num_params):
                    param_list.append(x_i[i][0])
                    x_i_new.append(x_i[i][1])
                x_i = np.array(x_i_new).T
                cv = LeaveOneOut()
                cv.get_n_splits(x_i)
                model = LinearRegression()
                test_error = []
                train_error = []
                for train_index, test_index in cv.split(x_i):
                    X_train, X_test = x_i[train_index], x_i[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model.fit(X_train, y_train)
                    y_hat = model.predict(X_test)
                    
                    test_error.append(abs(y_hat[0][0]-y_test[0][0]))
                    train_error.append(np.average(abs(y_train-model.predict(X_train))))

                test_error = np.array(test_error)
                train_error = np.array(train_error)
                test_error_mean = np.mean(test_error)
                test_error_std = np.std(test_error)
                train_error_mean = np.mean(train_error)
                train_error_std = np.std(train_error)
                
                if test_error_mean < test_error_min:
                    test_error_min = test_error_mean
                    print("Minimum\n\tTrain error {:.4f}+/-{:.5f}\n\tTest error {:.4f}+/-{:.5f}\n\tparam_list {}".format(train_error_mean, train_error_std, test_error_mean, test_error_std, self.ex_mat.columns[np.array(param_list)+1]))

#end def HeisenbergMapper

if __name__ == "__main__":
    structures = []
    energies   = []
    species    = []
    magmoms  = []
    magmoms_all  = []
    # Parse subdirectories. Each subdirectory should contain OUTCAR plus {vasprun.xml or CONTCAR}
    structures, energies, species, magmoms, magmoms_all = parse_dirs(structures, energies, species, magmoms, magmoms_all)
    print("Printing Structures")
    print(structures)
    print("End Printing Structures\n")
    interactions = {'Fe-Fe': {'nn': (3.1, 3.4)}}
    cutoff_nn = CutOffDictNN({('Fe', 'Fe'): 5.3})
    hm = HeisenbergMapper(structures, energies)
    hm.get_ex_mat(interactions, cutoff_nn)
    hm.fit()
    hm.report()
    #hm.plot()
