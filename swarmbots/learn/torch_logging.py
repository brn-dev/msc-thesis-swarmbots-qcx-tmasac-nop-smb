import torch


def enable_torch_compile_logging(verbose: bool = True):
    if verbose:
        torch._logging.set_logs(recompiles_verbose=True)
    else:
        torch._logging.set_logs(recompiles=True)
