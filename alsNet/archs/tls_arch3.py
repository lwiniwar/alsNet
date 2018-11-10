arch =[
    {
        'npoint': 8192,
        'radius': 0.05,
        'nsample': 16,
        'mlp': [128, 256, 256],
        'pooling': 'max_and_avg',
        'mlp2': None,
        'reverse_mlp': [256,256]
    },
    {
        'npoint': 4096,
        'radius': 0.15,
        'nsample': 16,
        'mlp': [128, 512, 512],
        'pooling': 'max_and_avg',
        'mlp2': None,
        'reverse_mlp': [512,512]
    },
    {
        'npoint': 2048,
        'radius': 0.5,
        'nsample': 16,
        'mlp': [256, 512, 512],
        'pooling': 'max_and_avg',
        'mlp2': None,
        'reverse_mlp': [512,512]
    },
    {
        'npoint': 512,
        'radius': 1,
        'nsample': 32,
        'mlp': [512, 1024, 1024],
        'pooling': 'max_and_avg',
        'mlp2': None,
        'reverse_mlp': [1024,1024]
    }, ]