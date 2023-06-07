import os
# for a in range(0,7):
#     s = a*30
#     e = (a+1)*30
#     command = 'python FusionHybridNetworkInput159.py --start {} --end {}'.format(s,e)
#     os.system(command)


for i in range(7):
    command = f'python Fusion.py train {i}'
    os.system(command)


