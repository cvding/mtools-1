import os
import torch
import numpy as np


class IOTensor:
    """save a numpy or torch tensor to file

        Args:
            tensor_path (str): the tensor path(*.np or *.npy)
    """
    def __init__(self, tensor_path=None):
        self.dicts = {}

        if tensor_path and os.path.exists(tensor_path):
            self.dicts = self.load(tensor_path)

    def __setitem__(self, key, val):
        if isinstance(val, (list, tuple)):
            vals = []
            for v in val:
                if isinstance(v, torch.Tensor):
                    vals.append(v.numpy())
                elif isinstance(v, np.ndarray):
                    vals.append(v)
                else:
                    raise ValueError('val[|unit] must be torch.Tensor or np.ndarray')
        else:
            if isinstance(val, torch.Tensor):
                vals = val.numpy()
            elif isinstance(val, np.ndarray):
                vals = val
            else:
                raise ValueError('val[|unit] must be torch.Tensor or np.ndarray')

        self.dicts[key] = vals
    
    def __getitem__(self, key):
        return self.dicts[key]
    
    def save(self, tensor_path):
        assert len(self.dicts) > 0, 'no data to save'
        bdir = os.path.dirname(tensor_path)
        if not os.path.exists(bdir):
            os.makedirs(bdir)
        
        np.save(tensor_path, self.dicts)
    
    def load(self, tensor_path):
        dpicle = np.load(tensor_path, allow_pickle=True)
        return dpicle.item()
    
    def save_to_part(self, tmp_dir='./temp'):
        """save tensor to part files

        Args:
            tmp_dir (str, optional): save data into temp path. Defaults to './temp'.

        Returns:
            dict: dict of tensor path 
        """
        assert len(self.dicts) > 0, 'no data to save'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        name_dict = {} 
        for key in self.dicts.keys():
            name_dict[key] = []
            for i, v in enumerate(self.dicts[key]):
                tensor_path = os.path.join(tmp_dir, '%s.%d.bin' % (key, i))
                v.tofile(tensor_path)
                name_dict[key].append(tensor_path)
        return name_dict
            
    @property
    def shape(self):
        def __shape(key):
            shapes = []
            if isinstance(self.dicts[key], (list, tuple)):
                for v in self.dicts[key]:
                    shapes.append(list(v.shape))
            else:
                shapes.append(list(self.dicts[key].shape))
            return shapes

        shapes = {}
        for key in self.dicts.keys():
            shapes[key] = __shape(key)
        
        return shapes

    @property 
    def range(self):
        def __range(key):
            ranges = []
            if isinstance(self.dicts[key], (list, tuple)):
                for v in self.dicts[key]:
                    ranges += [v.min(), v.max()]
            else:
                ranges += [self.dicts[key].min(), self.dicts[key].max()]
            return ranges

        ranges = {}
        for k in self.dicts.keys():
            ranges[k] = __range(k)
        
        return ranges
    
    def clean(self):
        self.dicts = {}