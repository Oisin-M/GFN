from torch import nn
from gfn_rom.gfn import GFN_AE

class GFN_ROM(nn.Module):
    
    def __init__(self, mesh_m, N_basis_factor, n_params, act, ae_sizes, map_sizes):
        super().__init__()
        gfn_latent_size = int(N_basis_factor*n_params)
        
        if len(ae_sizes)!=0:
            self.GFN = GFN_AE(mesh_m, ae_sizes[0])
        else:
            self.GFN = GFN_AE(mesh_m, gfn_latent_size)
        
        self.act = act()
        
        module_list_enc = []
        module_list_dec = []
        
        for i in range(1,len(ae_sizes)):
            module_list_dec.append(self.act)
            module_list_enc.append(nn.Linear(ae_sizes[i-1], ae_sizes[i]))
            module_list_dec.append(nn.Linear(ae_sizes[i], ae_sizes[i-1]))
            module_list_enc.append(self.act)
        if len(ae_sizes)!=0:
            module_list_dec.append(self.act)
            module_list_enc.append(nn.Linear(ae_sizes[-1], gfn_latent_size))
            module_list_dec.append(nn.Linear(gfn_latent_size, ae_sizes[-1]))
            module_list_enc.append(self.act)
        
        self.encoder = nn.Sequential(*module_list_enc)
        self.decoder = nn.Sequential(*module_list_dec[::-1])
        
        module_list_map = []
        
        for i in range(len(map_sizes)):
            if i==0:
                module_list_map.append(nn.Linear(n_params, map_sizes[i]))
            else:
                module_list_map.append(nn.Linear(map_sizes[i-1], map_sizes[i]))
            module_list_map.append(act())
        if len(map_sizes)!=0:
            module_list_map.append(nn.Linear(map_sizes[-1], gfn_latent_size))
            
        self.mapper = nn.Sequential(*module_list_map)
        
    def forward(self, x, mesh_n, params, update_master=False, expand=True, agglomerate=True, kd_tree_m=None, kd_tree_n=None, nn_m=None, nn_n=None):
        n_exp, n_agg = self.GFN.reshape_weights(mesh_n, update_master, expand, agglomerate, kd_tree_m, kd_tree_n, nn_m, nn_n)
        
        x_enc = self.act(x@self.GFN.We_n+self.GFN.be_m)
        x_enc = self.encoder(x_enc)
        
        x_map = self.act(self.mapper(params))
        
        x_recon = self.decoder(x_enc)
        x_recon = x_recon@self.GFN.Wd_n+self.GFN.bd_n
        
        return x_recon, x_enc, x_map, n_exp, n_agg