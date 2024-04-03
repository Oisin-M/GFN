import numpy as np
from sklearn import preprocessing
import torch

def train_test_split(params, n_splits, rate):
    num_graphs = params.shape[0]
    total_sims = int(num_graphs)
    rate = rate/100
    train_sims = int(rate * total_sims)
    test_sims = total_sims - train_sims
    main_loop = np.arange(total_sims).tolist()
    np.random.shuffle(main_loop)
    train_trajs = main_loop[0:train_sims]
    train_trajs.sort()
    test_trajs = main_loop[train_sims:total_sims]
    test_trajs.sort()
    train_trajs = np.array_split(train_trajs, n_splits)
    return train_trajs, test_trajs

def scaler_func():
    return preprocessing.StandardScaler()

def scaling(U):
    scaling_fun_1 = scaler_func()
    scaling_fun_2 = scaler_func()
    scaler_s = scaling_fun_1.fit(U)
    temp = torch.t(torch.tensor(scaler_s.transform(U)))
    scaler_f = scaling_fun_2.fit(temp)
    scaled_data = torch.unsqueeze(torch.t(torch.tensor(scaler_f.transform(temp))),0).permute(2, 1, 0)
    scale = [scaler_s, scaler_f]
    return scale, scaled_data[:,:,0]

def undo_scaling(U, scale):
    scaler_s = scale[0]
    scaler_f = scale[1]
    rescaled_data = torch.tensor(scaler_s.inverse_transform(torch.t(torch.tensor(scaler_f.inverse_transform(U.detach().numpy().squeeze())))))
    return rescaled_data

def get_scaled_data(fname):
    U_orig = torch.tensor(np.load("data/"+fname).T)
    scale, U_sc = scaling(U_orig)
    print('reconstruction error', ((U_orig - undo_scaling(U_sc, scale))**2).sum())
    return scale, U_sc

def load_and_process_datasets(train_fidelities=['large'], test_fidelities=['large', 'medium', 'small', 'tiny']):
    prefix_mesh = 'reference_mesh_'
    prefix_matrix = 'snapshot_matrix_'
    train_mesh_names = [prefix_mesh+''.join(f)+'.npy' for f in train_fidelities]
    train_solution_names = [prefix_matrix+''.join(f)+'.npy' for f in train_fidelities]
    test_mesh_names = [prefix_mesh+''.join(f)+'.npy' for f in test_fidelities]
    test_solution_names = [prefix_matrix+''.join(f)+'.npy' for f in test_fidelities]
    
    meshes_train = [np.load("data/"+i) for i in train_mesh_names]
    sols_train = [get_scaled_data(i)[1] for i in train_solution_names]
    # Assert that meshes and solutions have same numbers of points
    assert np.mean([meshes_train[i].shape[0] == sols_train[i].shape[1] for i in range(len(meshes_train))]) == 1
    
    meshes_test = [np.load("data/"+i) for i in test_mesh_names]
    sols_test = [get_scaled_data(i) for i in test_solution_names]
    # Assert that meshes and solutions have same numbers of points
    assert np.mean([meshes_test[i].shape[0] == sols_test[i][1].shape[1] for i in range(len(meshes_test))]) == 1
    
    return meshes_train, sols_train, meshes_test, sols_test
    
    