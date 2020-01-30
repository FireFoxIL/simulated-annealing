from matplotlib import pyplot as plt
from matplotlib import animation


def animate_history(country_map, history, step=10):
    fig = plt.figure()
    ims = []
    init_state = history[0]

    for i in range(0, len(init_state)):
        p = country_map[init_state[i]]
        plt.annotate(p.name, (p.geo_lon, p.geo_lat))

    for st in range(0, len(history), step):
        state = history[st]
        out_points_lons = []
        out_points_lats = []
        for i in range(0, len(state)):
            p1 = country_map[state[i-1]]
            p2 = country_map[state[i]]
            out_points_lons += [p1.geo_lon, p2.geo_lon]
            out_points_lats += [p1.geo_lat, p2.geo_lat]

        ims.append(plt.plot(out_points_lons, out_points_lats, '-ro'))

    anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=0)

    anim.save('animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()
