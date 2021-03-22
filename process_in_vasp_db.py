#!/usr/bin/env python
import pandas as pd
from pymatgen.analysis.local_env import CutOffDictNN
from pymatgen.analysis.magnetism.heisenberg import HeisenbergMapper
from os import sep
from pymatgen.core import Structure
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer
import pymongo
import numpy as np
from pprint import pprint
import pdb

client = pymongo.MongoClient("<<INPUT HERE>>")
db = client.vasp
col = db.magnetic_orderings

formula = "<<INPUT HERE>>"
items = col.find({"formula_pretty": formula})

structures = []
energies = []
print1 = []
print2 = []
print3 = []
item_num = 0
for item in items:
    item_num += 1
    structure = Structure.from_dict(item['structure'])
    magmom = np.round(structure.site_properties['magmom'])
    structure.add_site_property('magmom', magmom)
    mag_str = CollinearMagneticStructureAnalyzer(
        structure).get_structure_with_only_magnetic_atoms()
    magmom = mag_str.site_properties['magmom']
    species = mag_str.species
    print1.append(species)
    print2.append(magmom)
    print3.append(item['energy_per_atom'])
    structures.append(structure)
    energies.append(item['energy_per_atom'] * structure.num_sites)

print("There are {} calculations with the specified query in the database".format(item_num))
pprint(print1)
pprint(print2)
print(energies)


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
    def __init__(self, structures, energies, natoms_max):
        self.structures = structures
        self.energies = energies
        self.natoms_max = natoms_max

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
        pdb.set_trace()
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
        print(self.ex_params)
    
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
                    
try:
    interactions = {'Cr-Cr': {'nn': (2.8, 3.1), 'nnn': (3.1, 3.5)},
                    'Cr-Fe': {'nn': (2.8, 3.1), 'nnn': (3.1, 3.5)}}
    cutoff_nn = CutOffDictNN({('Cr', 'Cr'): 3.5, ('Cr', 'Fe'): 3.5})
    hm = HeisenbergMapper(structures, energies, 32)
    hm.get_ex_mat(interactions, cutoff_nn)
    hm.cross_validation()
    input("Cancel here and decide which interactions you want to keep. Then rerun the script erasing those interactions next")
    # Below I decided to erase Cr-Cr 2nd nearest neighbor interactions and then perform the fit using all data points
    intr_temp = interactions.copy()
    del intr_temp['Cr-Cr']['nnn']
    hm.get_ex_mat(intr_temp, cutoff_nn)
    hm.fit()
    hm.report()
    hm.plot()
except:
    print("Structure probably changes symmetry, interaction cutoffs may not be valid")
