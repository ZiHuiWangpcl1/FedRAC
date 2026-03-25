import random
import numpy as np
import torch
import math
import copy

def powerlaw(sample_indices, n_participants, alpha=1.65911332899, shuffle=False):
    # the smaller the alpha, the more extreme the division
    if shuffle:
        random.seed(1234)
        random.shuffle(sample_indices)

    from scipy.stats import powerlaw
    import math
    party_size = int(len(sample_indices) / n_participants)
    b = np.linspace(powerlaw.ppf(0.01, alpha),
                    powerlaw.ppf(0.99, alpha), n_participants)
    shard_sizes = list(map(math.ceil, b/sum(b)*party_size*n_participants))
    indices_list = []
    accessed = 0
    for participant_id in range(n_participants):
        indices_list.append(
            sample_indices[accessed:accessed + shard_sizes[participant_id]])
        accessed += shard_sizes[participant_id]
    return indices_list

def compute_grad_update(old_model, new_model, device=None):
    # maybe later to implement on selected layers/parameters
    origin_device = old_model.get_device()
    old_model, new_model = old_model.to(origin_device), new_model.to(origin_device)
    return [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]

def flatten(grad_update):
    return torch.cat([update.data.view(-1) for update in grad_update])

def unflatten(flattened, normal_shape):
    grad_update = []
    for param in normal_shape:
        n_params = len(param.view(-1))
        grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size()))
        flattened = flattened[n_params:]

    return grad_update

def mask_grad_update_by_magnitude(grad_update, mask_constant):
    grad_update = copy.deepcopy(grad_update)
    for i, update in enumerate(grad_update):
        grad_update[i].data[update.data.abs() < mask_constant] = 0
    return grad_update

#largest-values criterion
def mask_grad_update_by_order(grad_update, mask_order=None, mask_percentile=None, mode='all'):

	if mode == 'all':
		# mask all but the largest <mask_order> updates (by magnitude) to zero
		all_update_mod = torch.cat([update.data.view(-1).abs()
									for update in grad_update])
		if not mask_order and mask_percentile is not None:
			mask_order = int(len(all_update_mod) * mask_percentile)
		
		if mask_order == 0:
			return mask_grad_update_by_magnitude(grad_update, float('inf'))
		else:
			topk, indices = torch.topk(all_update_mod, mask_order)
			return mask_grad_update_by_magnitude(grad_update, topk[-1])

	elif mode == 'layer': # layer wise largest-values criterion
		grad_update = copy.deepcopy(grad_update)

		mask_percentile = max(0, mask_percentile)
		for i, layer in enumerate(grad_update):
			layer_mod = layer.data.view(-1).abs()
			if mask_percentile is not None:
				mask_order = math.ceil(len(layer_mod) * mask_percentile)

			if mask_order == 0:
				grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
			else:
				topk, indices = torch.topk(layer_mod, min(mask_order, len(layer_mod)-1))
				grad_update[i].data[layer.data.abs() < topk[-1]] = 0
		return grad_update

def add_gradient_updates(grad_update_1, grad_update_2, weight = 1.0):
    for param_1, param_2, j in zip(grad_update_1, grad_update_2, range(len(grad_update_1))):
        param_1 = param_1
        param_2 = param_2
        param_1.data += param_2.data * weight
    return grad_update_1

def add_update_to_model(model, update, weight=1.0, device=None):
    if not update: 
        return model

    for i in range(len(update)):
        update[i] = update[i].to(device)

    for param_model, param_update in zip(model.parameters(), update):
        param_model.data += weight * param_update.data
        
    return model

def add_gradients_to_model_batch(models, gradients, weights, device=None):
    '''
        Slightly faster than looping add_gradients_to_model
        
        models: list of N models
        gradients: list of N gradients
        weights: list of N x N, where weights[i] stores the dim N weights for each gradient
    '''
    
    state_dicts = [model.state_dict() for model in models]
    a = [model_.state_dict() for model_ in gradients]


    for i, state in enumerate(state_dicts):
        for j, gradient in enumerate(a):
            for k in state_dicts[0].keys():
                state[k] = state[k].float()
                gradient[k] = gradient[k].float()
                state[k] += gradient[k] * weights[i][j]

                aaaa = 0

    # for i, gradient in enumerate(a):
    #     for j, state in enumerate(state_dicts):
    #         for k in state_dicts[0].keys():
    #             state[k] = state[k].float()
    #             state[k] += gradient[k] * weights[j][i]

    for i, model in enumerate(models):
        model.load_state_dict(state_dicts[i])
    
    # a = [i for i in range(len(weights))]
    # for i, weight in zip(a, weights):
    #     for w, gradient in zip(weight, gradients):
    #         for old_param, new_param in zip(models[i].parameters(), gradient.parameters()):
    #             old_param.data += new_param.data * w

    bbb = 0
        
def add_gradients_to_model(model, gradients, weights, device=None):
    
    global_dict = model.state_dict()
    a = [model.state_dict() for model in gradients]
    
    for k in global_dict.keys():
        global_dict[k] = model.state_dict()[k].float()
        for i in range(len(gradients)):
            global_dict[k] += a[i][k] * weights[i]
  
    model.load_state_dict(global_dict)
