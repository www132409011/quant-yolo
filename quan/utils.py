import logging

from .func import *
from .quantizer import *
import numpy as np

def quantizer(default_cfg, this_cfg=None):
    target_cfg = dict(default_cfg)
    if this_cfg is not None:
        for k, v in this_cfg.items():
            target_cfg[k] = v

    if target_cfg['bit'] is None:
        q = IdentityQuan
    elif target_cfg['mode'] == 'lsq':
        q = LsqQuan
    else:
        raise ValueError('Cannot find quantizer `%s`', target_cfg['mode'])

    target_cfg.pop('mode')
    return q(**target_cfg)


def find_modules_to_quantize(model, quan_scheduler):
    replaced_modules = dict()
    for name, module in model.named_modules():
        if type(module) in QuanModuleMapping.keys():
            if name in quan_scheduler['excepts']:
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    quan_w_fn=quantizer(quan_scheduler['weight'],
                                        quan_scheduler['excepts'][name]['weight']),
                    quan_a_fn=quantizer(quan_scheduler['act'],
                                        quan_scheduler['excepts'][name]['act'])
                )
            else:
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    quan_w_fn=quantizer(quan_scheduler['weight']),
                    quan_a_fn=quantizer(quan_scheduler['act'])
                )
        elif name in quan_scheduler['excepts']:
            logging.warning('Cannot find module %s in the model, skip it' % name)

    return replaced_modules


def replace_module_by_names(model, modules_to_replace):
    def helper(child: t.nn.Module):
        for n, c in child.named_children():
            if type(c) in QuanModuleMapping.keys():
                for full_name, m in model.named_modules():
                    if c is m:
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)

    helper(model)
    return model

def inject_weight_variation(model, bit ,variation = 1./32.):
    def helper(child: t.nn.Module):
        for n, c in child.named_children():
            if type(c) in [QuanConv2d]:
                for full_name , m in model.named_modules():
                    if c is m:
                        if type(m.quan_w_fn) is IdentityQuan:
                            break
                        weight_base =m.weight.data.clone()
                        weight_base.div_(m.quan_w_fn.s).clamp_(-2 ** (bit -1), 2 ** (bit-1) -1).round_()
                        std =variation * (2**(bit -1))
                        noise = t.tensor(np.random.normal(0,std,m.weight.shape)).float().cuda()
                        weight_base.add_(noise)
                        weight_base.mul_(m.quan_w_fn.s)
                        m.weight.data.copy_(weight_base)
                        break
            else:
                helper(c)
    helper(model)
    return model