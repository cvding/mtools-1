import numpy as np

def diff(data1, data2, eps, bins=10):
    """calc diff between data1 and data2

    Args:
        data1 (np.ndarray): input data1 
        data2 (np.ndarray): input data2
        eps (float): threshold eps 
        bins (int or list, optional): num bins or list bin. Defaults to 10.

    Returns:
        dict: diff data
    """
    result = {}
    data1 = np.array(data1).flatten()
    data2 = np.array(data2).flatten()
    if len(data1)!=len(data2):
        print("error in [len(data1)!=len(data2)].\n"
            "data1.lens = {}, data2.lens = {}.".format(len(data1), len(data2)))
        return -1
    
    diff = data1-data2
    result['range'] = [diff.min().tolist(), diff.max().tolist()]
    diff = np.abs(diff)
    result['max_diff'] = diff.max().tolist()

    mask = diff > eps
    diff_count = mask.sum()
    avg_diff = diff.sum()/len(data1)
    result['avg_diff'] = avg_diff.tolist()
    result['num_diff'] = diff_count.tolist()
    hist, bins = np.histogram(diff, bins=bins)
    result['hist'] = list(zip(hist.tolist(), bins.tolist()))

    print(result)
    return result