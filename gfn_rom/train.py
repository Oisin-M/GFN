import numpy as np
import torch
from torch import nn
from pykdtree.kdtree import KDTree
from tqdm import tqdm

def train(model, opt, meshes_train, sols_train, params, train_trajs, test_trajs, update_master, epochs, mapper_weight, save_name):
    best_loss = np.inf
    test_losses = []
    train_losses = []

    # Find the initial nn lists
    kd_tree_m = KDTree(model.GFN.mesh_m)
    nn_ns = [kd_tree_m.query(mesh, k=1)[1].astype('int') for mesh in meshes_train]
    nn_ms = [KDTree(mesh).query(model.GFN.mesh_m, k=1)[1].astype('int') for mesh in meshes_train]

    check_if_expanded = True
    # List to check if we need to agglomerate or expand
    need_agg = [True]*len(meshes_train)
    need_exp = [True]*len(meshes_train)
    
    def criterion(x, x_recon, x_enc, x_map):
        return nn.functional.mse_loss(x, x_recon)+mapper_weight*nn.functional.mse_loss(x_enc, x_map)
    
    loop = tqdm(range(epochs))
    for i in loop:
        # Put in training mode and reset gradients
        model.train()
        opt.zero_grad()

        params_train = params[train_trajs[0]]
        n_expansions = 0
        
        loss = 0
        
        for j in range(len(meshes_train)):
            # Load data
            U_train = sols_train[j][train_trajs[j]]
            params_train = params[train_trajs[j]]
            mesh_n = meshes_train[j]
            # Predict
            x_recon, x_enc, x_map, n_exp, n_agg = model(U_train, mesh_n, params_train, update_master=update_master, nn_m=nn_ms[j], nn_n=nn_ns[j], expand=need_exp[j], agglomerate=need_agg[j])
            
            # If our master mesh is still changing, check if we had to expand or agglomerate this iteration
            # And update whether or not we will have to for future epochs accordingly
            if check_if_expanded:
                if n_exp == 0:
                    need_exp[j] = False
                if n_agg == 0:
                    need_agg[j] = False
            
            n_expansions += n_exp
            loss += criterion(U_train, x_recon, x_enc, x_map) * mesh_n.shape[0]
            
        # If we had an expansion, our master mesh changed and therefore we need to be be sure to expand and agglomerate
        # and also to update nearest neighbour lists due to this large master mesh
        # If there was no expansion, there will also be none going forward so we stop checking for any future epochs
        if n_expansions > 0 and check_if_expanded and update_master:
            kd_tree_m = KDTree(model.GFN.mesh_m)
            nn_ns = [kd_tree_m.query(mesh, k=1)[1].astype('int') for mesh in meshes_train]
            nn_ms = [KDTree(mesh).query(model.GFN.mesh_m, k=1)[1].astype('int') for mesh in meshes_train]
            need_agg = [True] * len(meshes_train)
            need_exp = [True] * len(meshes_train)
        else:
            check_if_expanded = False
        
        loss /= np.sum([k.shape[0] for k in meshes_train])
        loss.backward()
        opt.step()
        
        train_loss = loss.item()
        
        # Put model in evaluation mode for testing and don't track gradients here
        model.eval()

        test_loss = 0

        with torch.no_grad():
            params_test = params[test_trajs]

            for j in range(len(meshes_train)):
                U_test = sols_train[j][test_trajs]
                mesh_n = meshes_train[j]
                
                x_recon, x_enc, x_map, _, _ = model(U_test, mesh_n, params_test, update_master=update_master, nn_m=nn_ms[j], nn_n=nn_ns[j], expand=need_exp[j], agglomerate=need_agg[j])
                
                test_loss += criterion(U_test, x_recon, x_enc, x_map).item() * mesh_n.shape[0]
            
            test_loss /= np.sum([k.shape[0] for k in meshes_train])
        
        loop.set_postfix({"Loss(training)": train_loss, "Loss(testing)": test_loss})
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = i
            torch.save(model.state_dict(), "models/best_model_"+save_name+".pt")
            
    return train_losses, test_losses