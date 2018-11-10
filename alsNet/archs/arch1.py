arch =[
    {
        'npoint': 4096,
        'radius': 1,
        'nsample': 32,
        'mlp': [512, 512, 1024],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [128,128]
    },
    {
        'npoint': 2048,
        'radius': 5,
        'nsample': 64,
        'mlp': [128, 128, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    },
    {
        'npoint': 512,
        'radius': 15,
        'nsample': 64,
        'mlp': [128, 128, 256],
        'pooling': 'max',
        'mlp2': None,
        'reverse_mlp': [256,256]
    }, ]