import torch
import torch.nn as nn


def cuda(tensor, uses_cuda):#si uses_cuda est vrai, on retourne le tenseur sur le GPU
    return tensor.cuda() if uses_cuda else tensor



