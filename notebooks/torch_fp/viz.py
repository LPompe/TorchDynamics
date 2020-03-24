

"""
def update(i):
    angle = np.arange(360)[i]
    ax.view_init(15, angle)
    plt.draw()
    plt.pause(.00001)
    

anim = FuncAnimation(fig, update, frames=np.arange(0, 60), interval=0.01)
anim.save('/media/STORAGE10TB/lucas/line.gif', dpi=50, writer='imagemagick')
"""