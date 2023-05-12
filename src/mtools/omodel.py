import onnxruntime


class OnnxModel():
    """ constructor for OnnxModel

        Args:
            onnx_path (str): onnx model path 
    """
    def __init__(self, onnx_path):
       
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session=None):
        """get onnx model output name

        Args:
            onnx_session (OnnxInferenceSession, optional): Onnxruntime InferenceSession. Defaults to None.

        Returns:
            list: output name list
        """
        onnx_session = onnx_session or self.onnx_session
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session=None):
        """get onnx model input name

        Args:
            onnx_session (OnnxInferenceSession, optional): Onxxruntime InferenceSession. Defaults to None.

        Returns:
            list: output name list
        """
        onnx_session = onnx_session or self.onnx_session
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def __call__(self, input_dict):
        """run onnx model

        Args:
            input_dict (dict, list): input data, if list, the order must be same as input_name

        Returns:
            list: output data
        """

        input_dict = self.preproc(input_dict)
        if isinstance(input_dict, (list, tuple)):
            input_dict = dict(zip(self.input_name, input_dict))
        result = self.onnx_session.run(self.output_name, input_feed=input_dict)
        return self.postproc(result)
    
    def preproc(self, input_dict):
        return input_dict
    
    def postproc(self, result):
        return result