import json
import torch
from packaging import version
from pathlib import Path
from torch.onnx import export
import platform


is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")


def model_shape(model, model_args, input_names=None, output_names=None):
    shapes = {'input': {}, 'output': {}}
    if isinstance(model_args, dict):
        result = model(**model_args)
        for k, v in model_args.items():
            try:
                shapes['input'][k] = list(v.shape)
            except:
                shapes['input'][k] = str(type(v))
    elif isinstance(model_args, (tuple, list)):
        result = model(*model_args)
        if input_names is not None:
            assert len(input_names) == len(model_args), f"input_names should have {len(model_args)} elements, but got {len(input_names)}"
            for v, n in zip(model_args, input_names):
                try:
                    shapes['input'][n] = list(v.shape)
                except:
                    shapes['input'][n] = str(type(v))
        else:
            for i, v in enumerate(model_args):
                try:
                    shapes['input'][f'input_{i}'] = list(v.shape)
                except:
                    shapes['input'][f'input_{i}'] = str(type(v))
    else:
        result = model(model_args)
        if input_names is not None:
            assert len(input_names) == 1, f"input_names should have 1 element, but got {len(input_names)}"
            shapes['input'][input_names[0]] = list(model_args.shape)
        else:
            shapes['input']['input_0'] = list(model_args.shape)
    
    assert not isinstance(result, dict), "model output should not be dict"
    if isinstance(result, (tuple, list)):
        for i, v in enumerate(result):
            name = f'output_{i}' if output_names is None else output_names[i]
            try:
                shapes['output'][name] = list(v.shape)
            except:
                shapes['output'][name] = str(type(v))
    else:
        name = f'output_0' if output_names is None else output_names[0] 
        try:
            shapes['output']['output_0'] = list(result.shape)
        except:
            shapes['output']['output_0'] = str(type(result))
    
    return shapes


class Torch2Onnx:
    def __init__(self, opset=14, use_external_data_format=False) -> None:
        self.input_extra = {
            "opset": opset,
            "use_external_data_format": use_external_data_format,
        }
    
    def __parse_args(self, model_args):
        input = []
        names = []
        if isinstance(model_args, dict):
            for k, v in model_args.items():
                input.append(v)
                names.append(k)
        elif isinstance(model_args, (tuple, list)):
            input = model_args
            names = [f"input_{i}" for i in range(len(model_args))]
        else:
            raise TypeError(f"model_args should be dict, tuple or list, but got {type(model_args)}")
        return input, names
    
    def __call__(self, model, model_args, output_path, output_names:tuple=None, dynamic_axes:dict=None):
        input, names = self.__parse_args(model_args)

        self.onnx_export(model, input, output_path, names, output_names, dynamic_axes, **self.input_extra)  

        shapes = model_shape(model, model_args, names, output_names)

        with open(output_path + '.shape.json', 'w') as f:
            shapes['opset'] = self.input_extra['opset']
            shapes['python'] = platform.python_version()
            shapes['torch'] = torch.__version__
            json.dump(shapes, f, indent=4)

    @staticmethod
    def onnx_export(
        model,
        model_args: tuple,
        output_path: Path,
        ordered_input_names,
        output_names,
        dynamic_axes,
        opset,
        use_external_data_format=False,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # PyTorch deprecated the `enable_onnx_checker` and `use_external_data_format` arguments in v1.11,
        # so we check the torch version for backwards compatibility
        if is_torch_less_than_1_11:
            export(
                model,
                model_args,
                f=output_path.as_posix(),
                input_names=ordered_input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                use_external_data_format=use_external_data_format,
                enable_onnx_checker=True,
                opset_version=opset,
            )
        else:
            export(
                model,
                model_args,
                f=output_path.as_posix(),
                input_names=ordered_input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                opset_version=opset,
            )