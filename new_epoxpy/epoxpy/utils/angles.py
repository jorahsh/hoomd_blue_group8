import numpy as np


class Angles:
    @staticmethod
    def _filter_bonds(snapshot, types):
        filtered_bonds = []
        n_bonds = snapshot.bonds.N
        for bond_id1 in range(n_bonds):
            bg1 = snapshot.bonds.group[bond_id1]
            bg1p1 = bg1[0]
            bg1p2 = bg1[1]
            bg1p1_type = snapshot.particles.types[snapshot.particles.typeid[bg1p1]]
            bg1p2_type = snapshot.particles.types[snapshot.particles.typeid[bg1p2]]
            if bg1p1_type in types and bg1p2_type in types:#both types has to be in the triplet
                filtered_bonds.append(bg1)
        return filtered_bonds

    @staticmethod
    def _get_chains(bonds):
        chains = []
        prev_len = -1
        ii = 0
        while len(chains) != prev_len:
            prev_len = len(chains)
            chains = []
            for bg in bonds:
                if len(chains) == 0:
                    chains.append(set(bg))
                else:
                    found = False
                    found_index = -1
                    for i, chain in enumerate(chains):
                        if not chain.isdisjoint(set(bg)):
                            found = True
                            found_index = i
                            break
                    if found:
                        chains[found_index] = chains[found_index].union(set(bg))
                    else:
                        chains.append(set(bg))
            bonds = [tuple(chain) for chain in chains]
            if ii > 1e4:
                raise ValueError('This seems to have run into an infinite loop')
            else:
                ii += 1
        chain_list = [sorted(list(chain)) for chain in chains]
        return chain_list

    @staticmethod
    def _batch_gen(data, batch_size):
            for i in range(0, len(data), batch_size):
                            yield data[i:i+batch_size]

    def get_angles_for_linear_chains(self, snapshot, lin_chain_type, simple=False, lin_chain_N=10):
        '''
        Finds all triplets of particles that are of the given type into the angles list of the snapshot data.
        :param snapshot: HOOOMD shapshot to modify and return 
        :param types: A single particle type (e.g. 'A') 
        :return: list of particle id triplets of all the angles as a list of list
        '''
        if not isinstance(lin_chain_type, str):
            raise ValueError('lin_chain_type should be a string. You passed {}'.format(lin_chain_type))

        if simple:  # makes a further assumption that the linear chain has particles with sequential ids
            types = np.asarray(snapshot.particles.types)
            typeids = snapshot.particles.typeid
            lin_chain_typeid = np.where(types==lin_chain_type)[0]
            p_ids=np.where(typeids==lin_chain_typeid)[0]
            filtered_chains = [lin_chain for lin_chain in self._batch_gen(p_ids, lin_chain_N)]
        else:
            filtered_bonds = np.asarray(self._filter_bonds(snapshot, [lin_chain_type, lin_chain_type, lin_chain_type]))
            filtered_chains = self._get_chains(filtered_bonds)
        angle_triplets = []
        for CC_chain in filtered_chains:
            for first, second, third in zip(CC_chain, CC_chain[1:], CC_chain[2:]):
                angle_triplets.append([first, second, third])
        return angle_triplets
