arch =[
    {
        'npoint': 8192,
        'radius': 1,
        'nsample': 16,
        'mlp': [64, 512, 128],
        'pooling': 'max_and_avg',
        'mlp2': None,
        'reverse_mlp': [128,64]
    },
    {
        'npoint': 4096,
        'radius': 5,
        'nsample': 64,
        'mlp': [128, 512, 256],
        'pooling': 'max_and_avg',
        'mlp2': None,
        'reverse_mlp': [256,256]
    },
    {
        'npoint': 2048,
        'radius': 15,
        'nsample': 64,
        'mlp': [128, 512, 256],
        'pooling': 'max_and_avg',
        'mlp2': None,
        'reverse_mlp': [256,256]
    }, ]