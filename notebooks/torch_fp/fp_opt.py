import torch
import time

def vprint(t, v):
    if v:
        print(t)

        
fp_mse = torch.nn.MSELoss(reduction='none').cuda()
        
def find_fps(fun, candidates, params, x_star=None, verbose=True):
    """
    parameters:
        fun - function of which the fixed points will be approximated
        candidates - candidate fixed points
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
    v = verbose
    vprint("Optimising to find fixed points", v)
    
    if x_star is None:
        x_star = torch.zeros((candidates.shape[0], 1)).cuda()
        
    fp_candidates = torch.nn.Parameter(candidates.clone())
    
    fp_opt = torch.optim.Adam([fp_candidates], 
                         params['step_size'],
                         (params['adam_b1'],params['adam_b2']),
                         params['adam_eps'],
                         )
    
    
    
    start_time = time.time()
    
    n_candidates = fp_candidates.shape[0]
    batch_start = 0
    for i in range(0, params['num_batches']):
        
        
        fp_opt.zero_grad()
        
        batch_end = batch_start + params['batch_size']
        
            
        batch = fp_candidates[batch_start:batch_end]
        
        h_new = fun(x_star[batch_start:batch_end], batch)
        fp_loss = fp_mse(h_new, batch)

        mse = torch.mean(fp_loss)
        
        if mse < params['fp_opt_stop_tol']:
            print('Stopping tolerance {} reached'.format(params['fp_opt_stop_tol']))
            break
            

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
            
    fixed_points, indices = clean_fps(x_star, fun, fp_candidates, params)
    loss = torch.mean(fp_mse(fun(x_star[indices], fixed_points), fixed_points), dim=1)
    
    return fixed_points.detach(), indices, loss.detach() 

    
def clean_fps(x_star, fun, fps, params):
    
    kept_indcs = torch.arange(len(fps))
    speed_cond = check_speed(x_star, fun, fps, params['fp_tol'])
    kept_indcs = kept_indcs[speed_cond]
    

    
    
    
    return fps[kept_indcs], kept_indcs
    
def check_speed(x_star, fun, fps, tol):
    updated = fun(x_star, fps)
    speed = torch.mean(fp_mse(updated,fps), dim=1)
    return (speed > tol)
    
def check_outlier(fps, tol):
    return (fps > tol)
    
def check_unique(fps, tol):
    pass
    #todo
    