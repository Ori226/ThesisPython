import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ManualSoftware\\ffmpeg\\bin\\ffmpeg.exe'

FFMpegWriter = manimation.FFMpegWriter()
metadata = dict(title='Movie Test', artist='Matplotlib',
        comment='Movie support!')
writer =  manimation.FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()
l, = plt.plot([], [], 'k-o')

plt.xlim(-5, 5)
plt.ylim(-5, 5) 

x0,y0 = 0, 0

with writer.saving(fig, "c:\\temp\\writer_test2.mp4", 100):
    for i in range(100):
        print(i)
        x0 += 0.1 * np.random.randn()
        y0 += 0.1 * np.random.randn()
        l.set_data(x0, y0)
        writer.grab_frame()