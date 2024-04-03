import numpy as np
import torch
from torch import nn
import math
from pykdtree.kdtree import KDTree

class GFN_AE(nn.Module):
    """
    Module implementing the GFN method for the encoder and decoder architectures.
    """
    def __init__(self, mesh_m, latent_size=20):
        super().__init__()
        size = mesh_m.shape[0]
        self.latent_size = latent_size
        
        # Set up master mesh and master weights
        # NB: cheaper not to make parameters if we're planning on updating sizes
        self.We_m = nn.Parameter(torch.empty(size, self.latent_size))
        self.be_m = nn.Parameter(torch.empty(self.latent_size))
        self.Wd_m = nn.Parameter(torch.empty(self.latent_size, size))
        self.bd_m = nn.Parameter(torch.empty(size))
        self.mesh_m = mesh_m
        
        # Initialise weights
        self.initialise(self.We_m, self.be_m)
        self.initialise(self.Wd_m, self.bd_m)
        
        # Set up the weight matrices and mesh used for inference
        # Note: no self.be_n since we never need to reshape the encoder biases i.e. be_n == be_m in all cases
        self.We_n = self.We_m.clone()
        self.Wd_n = self.Wd_m.clone()
        self.bd_n = self.bd_m.clone()
        
    def initialise(self, weight, bias):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
        bias.data.uniform_(-stdv, stdv)
        
    def reshape_weights(self, mesh_n, update_master=False, expand=True, agglomerate=True, kd_tree_m=None, kd_tree_n=None, nn_m=None, nn_n=None):
        """
        mesh_n: New geometry to evaluate on
        update_master: Whether or not to update the master mesh during the reshaping. Must always
                        be False during inference. For 'fixed' or 'preadapt' methods, should be
                        False. For 'adapt', should be True during training.
        expand: Whether or not the new geometry requires expanding the master mesh
        agglomerate: Whether or not the new geometry requires agglomerating the master mesh
        kd_tree_m: A precomputed kdtree using nodes in the master mesh
        kd_tree_n: A precomputed kdtree using nodes in the new mesh
        nn_m: List of nearest neighbours for nodes in the master mesh (i.e. indices of nodes in the new mesh)
        nn_n: List of nearest neighbours for nodes in the new mesh (i.e. indices of nodes in the master mesh)
        """
        
        # Find nearest neighbours if we don't already know them
        if nn_n is not None:
            pass
        elif kd_tree_n is not None:
            nn_n = kd_tree_m.query(mesh_n, k=1)[1]
        else:
            kd_tree_m = KDTree(self.mesh_m)
            nn_n = kd_tree_m.query(mesh_n, k=1)[1]
        if nn_m is not None:
            pass
        elif kd_tree_n is not None:
            nn_m = kd_tree_n.query(self.mesh_m, k=1)[1]
        else:
            kd_tree_n = KDTree(mesh_n)
            nn_m = kd_tree_n.query(self.mesh_m, k=1)[1]

        # WE HAVE TWO CASES TO CONSIDER:
        # Case 1: We are either in evaluation mode or we are in training with `update_master=False`
        # Case 2: We are in training with `update_master=True`
        
        # How much did we expand or agglomerate?
        nodes_added = 0
        nodes_combined = 0

        # !! EXPANSION !!
        if expand:
            # Case 1: master mesh isn't updated
            if not update_master or not self.training:
                # Set up intermediate variable so we do not update We_m
                if self.training:
                    We_i = self.We_m.clone()
                    Wd_i = self.Wd_m.clone()
                    bd_i = self.bd_m.clone()
                else:
                    We_i = self.We_m
                    Wd_i = self.Wd_m
                    bd_i = self.bd_m
            
                # Set up counters
                count_m = np.zeros(self.mesh_m.shape[0])
                size = self.mesh_m.shape[0]

                # Loop over each pt_n_{M_n} with its pair pt_m_{M_m} <- pt_n_{M_n}
                for pt_n, pt_m in enumerate(nn_n):
                    # if NOT pt_m_{M_m} <-> pt_n_{M_n}
                    if nn_m[pt_m] != pt_n:
                        nodes_added += 1
                        count_m[pt_m] += 1

                        # Divide encoder weights by number of expansions
                        We_i[pt_m] *= count_m[pt_m]/(count_m[pt_m]+1)
                        # Store the index of the weight we want
                        # so we can update at the end without storing
                        # all the nodes to update explictly
                        new_row = torch.zeros(1, We_i.shape[1], device=We_i.device)
                        new_row[0][0] = pt_m
                        We_i = torch.cat((We_i, new_row), dim=0)
                        
                        # Duplicate weights for decoder
                        Wd_i = torch.cat((Wd_i, Wd_i[:, pt_m:pt_m+1]), dim=1)
                        bd_i = torch.cat([bd_i, bd_i[pt_m:pt_m+1]])

                        # Directly add nearest neighbour link
                        nn_m = np.append(nn_m, pt_n)
                
                # Loop over the nodes we need to update using the index we stored in the first element
                for i in range(size, size+nodes_added):
                    index = int(We_i[i,0])
                    We_i[i] = We_i[index]

            # Case 2: master mesh is updated (only happens if model.train())
            else:
                # We overwrite master mesh so we can ignore gradient tracking
                with torch.no_grad():
                    # Set up counters
                    count_m = np.zeros(self.mesh_m.shape[0])
                    nodes_added = 0
                    size = self.mesh_m.shape[0]

                    # Loop over each pt_n_{M_n} with its pair pt_m_{M_m} <- pt_n_{M_n}
                    for pt_n, pt_m in enumerate(nn_n):
                        # if NOT pt_m_{M_m} <-> pt_n_{M_n}
                        if nn_m[pt_m] != pt_n:
                            nodes_added += 1
                            count_m[pt_m] += 1

                            # Divide encoder weights by number of expansions
                            self.We_m[pt_m] *= count_m[pt_m]/(count_m[pt_m]+1)
                            # Store the index of the weight we want
                            # so we can update at the end without storing
                            # all the nodes to update explictly
                            new_row = torch.zeros(1, self.We_m.shape[1], device=self.We_m.device)
                            new_row[0][0] = pt_m
                            self.We_m = torch.nn.Parameter(torch.cat((self.We_m, new_row), dim=0))
                            
                            # Duplicate weights for decoder
                            self.Wd_m = torch.nn.Parameter(torch.cat((self.Wd_m, self.Wd_m[:, pt_m:pt_m+1]), dim=1))
                            self.bd_m = torch.nn.Parameter(torch.cat([self.bd_m, self.bd_m[pt_m:pt_m+1]]))

                            # Directly add nearest neighbour link
                            nn_m = np.append(nn_m, pt_n)
                            
                            # Update master mesh
                            self.mesh_m = np.vstack([self.mesh_m, mesh_n[pt_n]])
                    
                    # Loop over the nodes we need to update using the index we stored in the first element
                    for i in range(size, size+nodes_added):
                        index = int(self.We_m[i,0])
                        self.We_m[i] = self.We_m[index]

                We_i = self.We_m.clone()
                Wd_i = self.Wd_m.clone()
                bd_i = self.bd_m.clone()

        # If we don't expand
        else:
            if self.training:
                We_i = self.We_m.clone()
                Wd_i = self.Wd_m.clone()
                bd_i = self.bd_m.clone()
            else:
                We_i = self.We_m
                Wd_i = self.Wd_m
                bd_i = self.bd_m

        # !! AGGLOMERATION !!
        if agglomerate:
            self.We_n = torch.zeros((mesh_n.shape[0], self.We_m.shape[1]), device=self.We_m.device)
            self.Wd_n = torch.zeros((self.We_m.shape[1], mesh_n.shape[0]), device=self.We_m.device)
            self.bd_n = torch.zeros(mesh_n.shape[0], device=self.We_m.device)

            count_n = np.zeros(mesh_n.shape[0])

            # Loop over each pt_m_{M_m} with its pair pt_m_{M_m} -> pt_n_{M_n}
            for pt_m, pt_n in enumerate(nn_m):

                count_n[pt_n]+=1
                if count_n[pt_n]>1:
                    nodes_combined+=1

                # Summation
                self.We_n[pt_n] += We_i[pt_m]
                # Calculate Mean
                self.Wd_n[:, pt_n] = ( (count_n[pt_n]-1)*self.Wd_n[:, pt_n] + Wd_i[:, pt_m] )/count_n[pt_n]
                self.bd_n[pt_n] = ( (count_n[pt_n]-1)*self.bd_n[pt_n] + bd_i[pt_m] )/count_n[pt_n]

        # If we don't agglomerate        
        else:
            self.We_n = We_i[nn_m]
            self.Wd_n = Wd_i[:,nn_m]
            self.bd_n = bd_i[nn_m]
            
        return nodes_added, nodes_combined