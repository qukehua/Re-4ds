import torch
dict = torch.load('ckpt/ckpt5.pth.tar', map_location='cpu')
print(type(dict))
for k in dict.keys():
    print(k)
print(len(dict['state_dict']))
print(type(dict['state_dict']))
for k, v in dict['state_dict'].items():
    print(k)