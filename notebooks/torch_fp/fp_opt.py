import torch
import time
import numpy as np

from scipy.spatial.distance import pdist, squareform


def vprint(t, v):
    if v:
        print(t)

        
fp_mse = torch.nn.MSELoss(reduction='none').cuda()
def find_fps(fun, fp_candidates, params, x_star=None, verbose=True, device='cpu'):
    """
    parameters:
        fun - function of which the fixed points will be approximated
        candidates - candidate fixed points, will be edited in place
        params - optimisation parameter dict, example: 
        
        {
            num_batches = 1400   # Total number of batches to train on
            batch_size = 128          # How many examples in each batch
            step_size = 0.2          # initial learning rate
            decay_factor = 0.9999     # decay the learning rate this much
            decay_steps = 1           #
            adam_b1 = 0.9             # Adam parameters
            adam_b2 = 0.999
            adam_eps = 1e-5
            opt_print_every = 200   # Print training information interval

        }
        
        x_star - input to fun, will be assumed 0 if not provided
        verbose - print info or not
        
    """
    assert type(fp_candidates) == torch.nn.Parameter, "candidates must be torch.nn.Parameter type (also clone)"
    
    v = verbose
    vprint("Optimising to find fixed points", v)
    
    if x_star is None:
        x_star = torch.zeros((fp_candidates.shape[0], 1)).to(device)
        
   
    
    fp_opt = torch.optim.Adam([fp_candidates],
                         params['step_size'],
                         (params['adam_b1'],params['adam_b2']),
                         params['adam_eps'],
                         )
    
    lr_lambda = lambda x: params['decay_factor']
    lr_annealler = torch.optim.lr_scheduler.MultiplicativeLR(fp_opt, lr_lambda, last_epoch=-1)
    
    
    
    start_time = time.time()
    
    n_candidates = fp_candidates.shape[0]
    batch_start = 0
    vprint(f"Optimising to find {n_candidates} fixed points", v)
    
    
    
    for i in range(0, params['num_batches']):
        
        
        if i > 0 and batch_start == 0:
            lr_annealler.step()
        
        batch_start = np.random.randint(0, n_candidates - params['batch_size'])
        
        fp_opt.zero_grad()
        
        batch_end = batch_start + params['batch_size']
        
            
        batch = fp_candidates[batch_start:batch_end].to(device)
        
        h_new = fun(x_star[batch_start:batch_end], batch)
        fp_loss = fp_mse(h_new, batch)

        mse = torch.mean(fp_loss)
        if mse < params['fp_opt_stop_tol']:
            continue
        
        #if mse_mean < params['fp_opt_stop_tol']:
        #    print('Stopping tolerance {} reached'.format(params['fp_opt_stop_tol']))
        #    break
            

        mse.backward()
        fp_opt.step()
        
        
        
        batch_time = time.time() - start_time
        start_time = time.time()
        if i % params['opt_print_every'] == 0:
            lr = lr_annealler.get_lr()[0]
            vprint("Batches {}-{} in {:0.2f}, Training loss {:0.4f}, LR {:0.4f}".format(batch_start, batch_end, batch_time, mse, lr), v)
            
    fixed_points = clean_fps(x_star, fun, fp_candidates, params)
    loss = torch.mean(fp_mse(fun(x_star[:len(fixed_points)], fixed_points), fixed_points), dim=1)
    
    return fp_candidates.detach(), loss

    
def clean_fps(x_star, fun, fps, params):
    
    kept_idcs = torch.arange(len(fps))
    speed_idcs = check_speed(x_star, fun, fps, params['fp_tol'])
    l_speed, l_total = sum(speed_idcs), len(fps)
    print(f"kept {l_speed}/{l_total} with speed check")

    kept_idcs = kept_idcs[speed_idcs]
    kept_fps = fps[speed_idcs]

    unique_idcs = check_unique(kept_fps, params['unique_tol'])
    l_unq = sum(unique_idcs)
    print(f"kept {l_unq}/{l_speed} with uniqueness check")
    
    kept_idcs = kept_idcs[unique_idcs]
    kept_fps = kept_fps[unique_idcs]

    outlier_idcs = check_outlier(kept_fps, params['outlier_tol'])
    l_outlier = sum(outlier_idcs)
    kept_idcs = kept_idcs[outlier_idcs]
    kept_fps = kept_fps[outlier_idcs]
    print(f"kept {l_outlier}/{l_unq} with outlier check")
    
    
    return kept_fps
    
def check_speed(x_star, fun, fps, tol):
    updated = fun(x_star, fps)
    speed = torch.mean(fp_mse(updated,fps), dim=1)
    return speed < tol
    
def check_unique(fps, tol):
    
    nfps = fps.shape[0]
    example_idxs = torch.arange(nfps)
    all_drop_idxs = []
  
    distances = torch.cdist(fps, fps)
    
    for fidx in range(nfps-1):
        distances_f = distances[fidx, fidx+1:]
        drop_idxs = example_idxs[fidx+1:][distances_f <= tol]
        all_drop_idxs.append(drop_idxs)

    unique_dropidxs = np.unique(torch.cat(all_drop_idxs))
    keep_idcs = np.setdiff1d(example_idxs, unique_dropidxs)
    keep_bool = torch.zeros(fps.shape[0], dtype=bool)
    keep_bool[keep_idcs] = True
    #if keep_idxs.shape[0] > 0:
        #unique_fps = fps[keep_idxs, :]
    #else:
        #unique_fps = onp.array([], dtype=onp.int64)

    return keep_bool
    
def check_outlier(fps, tol):
    
    distances = torch.cdist(fps, fps)
    
    nn, _ = torch.topk(distances, 1, largest=False)
    
    return (nn < tol).squeeze()
    
    
