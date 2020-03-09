import torch
import time

from scipy.spatial.distance import pdist, squareform

def vprint(t, v):
    if v:
        print(t)

        
fp_mse = torch.nn.MSELoss(reduction='none').cuda()
        git
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
    
    
    
    start_time = time.time()
    
    n_candidates = fp_candidates.shape[0]
    batch_start = 0
    mse_mean = 1
    for i in range(0, params['num_batches']):
        
        
        fp_opt.zero_grad()
        
        batch_end = batch_start + params['batch_size']
        
            
        batch = fp_candidates[batch_start:batch_end].to(device)
        
        h_new = fun(x_star[batch_start:batch_end], batch)
        fp_loss = fp_mse(h_new, batch)

        mse = torch.mean(fp_loss)
        mse_mean += mse / i
        
        #if mse_mean < params['fp_opt_stop_tol']:
        #    print('Stopping tolerance {} reached'.format(params['fp_opt_stop_tol']))
        #    break
            

        mse.backward()
        fp_opt.step()
        
        
        batch_time = time.time() - start_time
        start_time = time.time()
        if i % params['opt_print_every'] == 0:
            vprint("Batches {}-{} in {:0.2f}, Training loss {:0.4f}".format(batch_start, batch_end, batch_time, mse), v)
            
        if batch_end > n_candidates:
            batch_start = 0
        else:
            batch_start = batch_end
            
    fixed_points = clean_fps(x_star, fun, fp_candidates, params)
    loss = torch.mean(fp_mse(fun(x_star[indices], fixed_points), fixed_points), dim=1)
    
    return fixed_points.detach(), indices, loss.detach() 

    
def clean_fps(x_star, fun, fps, params):
    
    kept_indcs = torch.arange(len(fps))
    speed_idcs = check_speed(x_star, fun, fps, params['fp_tol'])
    kept_idcs = kept_idcs[speed_idcs]
    
    
    speed_idcs = check_unique(kept_fps, params['unique_tol'])
    kept_idcs = kept_idcs[speed_idcs]
    
    outlier_fps = check_outlier(kept_fps, params['fp_outlier_tol'])
    kept_idcs = kept_idcs[outlier_fps]
    
    
    return fps[kept_indcs]
    
def check_speed(x_star, fun, fps, tol):
    updated = fun(x_star, fps)
    speed = torch.mean(fp_mse(updated,fps), dim=1)
    return torch.where(speed > tol)
    
def check_unique(fps, loss_fun, tol):
    
    nfps = fps.shape[0]
    example_idxs = torch.arange(nfps)
    all_drop_idxs = []
  
    distances = torch.cdist(fps, fps)
    
    for fidx in range(nfps-1):
        distances_f = distances[fidx, fidx+1:]
        drop_idxs = example_idxs[fidx+1:][distances_f <= identical_tol]
        all_drop_idxs += list(drop_idxs)
        
    unique_dropidxs = np.unique(all_drop_idxs)
    keep_idcs = np.setdiff1d(example_idxs, unique_dropidxs)
    
    #if keep_idxs.shape[0] > 0:
        #unique_fps = fps[keep_idxs, :]
    #else:
        #unique_fps = onp.array([], dtype=onp.int64)
    
    return keep_idcs
    
def check_outlier(fps, tol):
    
    distances = torch.cdist(fps, fps)
    
    nn, _ = torch.topk(distances, 1)
    
    keep_idcs = torch.where(nn < tol)
    
    return keep_idcs
    
    
