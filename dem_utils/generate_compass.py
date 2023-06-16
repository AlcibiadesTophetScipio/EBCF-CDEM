import imageio.v2 as imageio
import numpy as np

'''
10.0, red
20.0, orange
30.0, yellow
40.0, green
50.0, cyan
60.0, aqua
70.0, blue
80.0, purple
90,0, red
'''


compass_color = [np.ones([10,10])*i*10 for i in range(1,10)]
compass_color = np.concatenate(compass_color, axis=0)
print(compass_color.shape)
print(set(compass_color.flatten().tolist()))
imageio.imsave('./results/compass_color.tif', compass_color)