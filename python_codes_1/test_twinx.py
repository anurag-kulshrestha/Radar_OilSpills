import numpy as np
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
t = np.arange(100).reshape(10,10)

inc_angle=np.arange(40,42,.2)
print(inc_angle)
#s1 = np.exp(t)
#ax1.plot(t, s1, 'b-')
im=ax1.imshow(t)
ax1.set_xlabel('Range')
ax1.set_ylabel('Azimuth')
fig.colorbar(im)
# Make the y-axis label, ticks and tick labels match the line color.
#ax1.set_ylabel('exp', color='b')
#ax1.tick_params('y', colors='b')
fig.colorbar(im, orientation='horizontal')
'''
ax2 = ax1.twinx()
#s2 = np.sin(2 * np.pi * t)
ax2.plot(np.arange(10),inc_angle, 'r.')
ax2.set_ylabel('INC_angle', color='r')
ax2.set_yticks(inc_angle)
#ax2.tick_params(inc_angle, colors='r')
'''


#fig.tight_layout()
plt.show()
