from torch.optim import Adam, AdamW

def get_optimizer(models, cfg):

    if not isinstance(models, list):
        models = [models]

    params = [m.parameters() for m in models]
    if cfg.optimizer == 'adam':
        optimizer = Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adamW':
        optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    optimizer.zero_grad()
    
    return optimizer