import numpy as np

file = open('logs/11_12_1.txt')
lines = file.readlines()
print(len(lines))
err1, err3, err7, err9 = [], [], [], []
t_n = 0
for l in range(len(lines)):
    line = lines[l]
    if 'average' in line:
        t_n += 1
        err1.append(float(lines[l + 1][:7]))
        err3.append(float(lines[l + 2][:7]))
        err7.append(float(lines[l + 3][:7]))
        err9.append(float(lines[l + 4][:7]))
print('Err1: {} {}'.format(min(err1), err1.index(min(err1)) * 80 + 77))
print('Err3: {} {}'.format(min(err3), err3.index(min(err3)) * 80 + 78))
print('Err7: {} {}'.format(min(err7), err7.index(min(err7)) * 80 + 79))
print('Err9: {} {}'.format(min(err9), err9.index(min(err9)) * 80 + 80))
print('{} {} {} {}'.format(min(err1), min(err3), min(err7), min(err9)))
