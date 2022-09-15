import torch
inpt = torch.rand((16, 66, 10))
print(inpt.shape)
print(inpt[0, 0])
out = inpt[:, :, -10:]
last_frame = inpt[:, :, -1]
print(last_frame.shape)
print(last_frame)
last_frame_seq = last_frame.repeat(10, 1, 1).permute(1, 2, 0)
print(last_frame_seq)
print(last_frame_seq.shape)
print(last_frame_seq[0, 0])