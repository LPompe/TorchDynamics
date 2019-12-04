import torch
import time

def find_fps(fun, candidates, params, x_star=None, verbose=True):
    """
    parameters:
        fun - function of which the fixed points will be approximated
        candidates - candidate fixed points
        params - optimisation parameter dict, example: 
        
        {
            fp_num_batches = 1400   # Total number of batches to train on
            fp_batch_size = 128          # How many examples in each batch
            fp_step_size = 0.2          # initial learning rate
            fp_decay_factor = 0.9999     # decay the learning rate this much
            fp_decay_steps = 1           #
            fp_adam_b1 = 0.9             # Adam parameters
            fp_adam_b2 = 0.999
            fp_adam_eps = 1e-5
            fp_opt_print_every = 200   # Print training information interval

        }
        
        x_star - input to fun, will be assumed 0 if not provided
        verbose - print info or not
        
    """
    if not x_star is None:
        x_star = torch.zeros((candidates.shape[0], 1)).cuda()
        
    fp_candidates = torch.nn.Parameter(candidates.clone())
    
    fp_opt = torch.optim.Adam([fp_candidates], 
                         params['fp_step_size'],
                         (params['fp_adam_b1'],params['fp_adam_b2']),
                         params['fp_adam_eps'],
                         )
    
    fp_mse = torch.nn.MSELoss(reduction='none').cuda()
    
    start_time = time.time()
    
    n_candidates = fp_candidates.shape[0]
    batch_start = 0
    for i in range(params['fp_num_batches']):
        
        
        fp_opt.zero_grad()
        
        batch_end = batch_start + params['fp_batch_size']
        
            
        batch = fp_candidates[batch_start:batch_end]
        
        h_new = fun(x_star[batch_start:batch_end], batch)
        fp_loss = fp_mse(h_new, batch)

        mse = torch.mean(fp_loss)

        mse.backward()
        fp_opt.step()
        
        if batch_end > n_candidates:
            batch_start = 0
        else:
            batch_start = batch_end

        if i % fp_opt_print_every == 0 and verbose:
            print(mse)
            
        return fp_candidates, fp_losses, fp_idxs, fp_opt_details

    
def clean_fps(fps, params)