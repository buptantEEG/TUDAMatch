def adamatch_hyperparams(lr=0.0001,lr_trg=1e-6,lr_dis=1e-2, tau=0.9, wd=3e-4, scheduler=False):
    """
    Return a dictionary of hyperparameters for the AdaMatch algorithm.

    Arguments:
    ----------
    lr: float
        Learning rate.s

    tau: float
        Weight of the unsupervised loss.

    wd: float
        Weight decay for the optimizer.

    scheduler: bool
        Will use a StepLR learning rate scheduler if set to True.

    Returns:
    --------
    hyperparams: dict
        Dictionary containing the hyperparameters. Can be passed to the `hyperparams` argument on AdaMatch.
    """
    
    hyperparams = {'learning_rate': lr,
                    'learning_rate_target': lr_trg,
                    'learning_rate_discriminator': lr_dis,
                   'tau': tau,
                   'weight_decay': wd,
                   'step_scheduler': scheduler
                   }

    return hyperparams