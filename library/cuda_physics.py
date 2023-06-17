from plots import plot_heatmap, plot_surface
import pandas as pd
import torch as t
import time, datetime

# profile the code with cProfile
import cProfile
import pstats

from tqdm import tqdm
from copy import deepcopy

template = "plotly_white"

def time_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# # 2 slits
# edges_slits = [[-25, -5], [5, 25]]  # two slits
# barrier_points = [[-i, i] for i in range(26, 35)] + [[-i, i] for i in range(1, 4)]
# o_x = t.tensor(edges_slits + barrier_points).reshape(1, -1).squeeze()

# 1 slit
# TODO: there is something wrong when the edges are not symmetrical
# TODO: create notebook with this code
edges_slits = [[-40, 0]]  # 1 slit
# edges_slits = [[-10, 10]]  # 1 slit
multiplier = 1  # was 10
barrier_points = [[-i / multiplier, i / multiplier] for i in range(int(10 * multiplier), int(20 * multiplier))]
barrier_points = [[-35, 35]]
# barrier_points = [[-10, 10]]
o_x = t.tensor(barrier_points).reshape(1, -1).squeeze()


device = t.device("mps")

# # edge diffraction
# edges_slits = [[-50, 10]] # 1 slit
# barrier_points = []
# o_x = t.tensor([10])

n_slits = len(edges_slits)
n_particles = 100  # <3 3080Ti
barrier_y = 0
dt = 1
v_y = 3  # FIX
x = 50
y = 30  # 10000
type_trig = "sin"
# G = -0.6626  # why is this negative?
G = 2
mass = 1  # FIX
wavelength = 0.1
power = 2
inner_power = 2
round_x = 5
n_bins = 200  # always odd numbers
PLOT_3D = True
PLOT_Z_DISTRIBUTION = True
PLOT_DENSITY = True
normalize = True
moving_wave = False
gravity_threshold = 500

p_x = t.Tensor()

for slit in edges_slits:
    p_x = t.cat((p_x, t.linspace(slit[0], slit[1], n_particles + 1)), 0)
    # remove the particles that would be superpositioned with edges
    # p_x = p_x[1:-1]
p_y = t.zeros_like(p_x)
p_xy = t.stack((p_x, p_y), 1).to(device)

# Let's try to keep the trail on the cpu
trail_xy = t.clone(p_xy.unsqueeze(0))
v_xy = t.zeros_like(p_xy).to(device) + t.tensor((0, v_y)).to(device)

p_mass = t.ones_like(p_y).to(device)
o_y = t.zeros_like(o_x) + barrier_y
o_xy = t.stack((o_x, o_y), 1).to(device)

# Stupid work around, code is becoming cluttered
dist = t.Tensor().to(device)
hist = t.histc(trail_xy[0][:, 0].to('cpu'), bins=n_bins, min=-x, max=x)
dist = t.cat((dist, hist.unsqueeze(0).to(device)), 0)

t_ = 0
for i in tqdm(range(y)):
    if i <= gravity_threshold:  
        dt_ = t.ones_like(p_y).unsqueeze(1).to(device) * dt
        a_xy = t.zeros_like(p_xy).to(device)
        for o in o_xy:
            d_xy = p_xy - o
            r = t.sqrt(t.sum(d_xy ** 2, axis=1))
            r_wavelength = r ** inner_power * wavelength
            F = G * mass / r ** 2
            if type_trig == "sin":
                F *= t.sin(r_wavelength - v_y * t_ * moving_wave) ** power
            elif type_trig == "cos":
                F *= t.cos(r_wavelength - v_y * t_ * moving_wave) ** power
            elif type_trig == "tan":
                F *= t.tan(r_wavelength - v_y * t_ * moving_wave) ** power
            # else:
            #     F *= r_wavelength - v_y * t_ * moving_wave
            a_xy += F.unsqueeze(1) * d_xy / t.stack([r, r], axis=1)
        v_xy += a_xy * dt_
        # calculate the velocity of each particle
        v = t.sqrt(t.sum(v_xy ** 2, axis=1))
        # clip the velocity to the max velocity which is v_y 
        v = t.clip(v, 0, v_y)
        # normalize the velocity vector
        v_xy = v_xy / t.stack([v, v], axis=1)
        # multiply the velocity vector by the max velocity
        v_xy *= v_y
    
        dt_ = (1 / v_xy)[:, 1].unsqueeze(1)
        t.abs_(dt_)
    p_xy += v_xy * dt_
    if round_x > 0:
        p_xy = p_xy.to('cpu').round(decimals=round_x).to(device)
    trail_xy = t.cat((trail_xy, p_xy.unsqueeze(0)), 0)
    hist = t.histc(trail_xy[i][:, 0].to('cpu'), bins=n_bins, min=-x, max=x).to(device)
    dist = t.cat((dist, hist.unsqueeze(0)), 0)
    t_ += dt

print("Creating particle grid")
# trails = trail_xy.reshape(-1, 2).cpu().numpy().round(round_x)
dist = pd.DataFrame(dist.cpu().numpy())
dist = dist.div(dist.max().max())
if round_x > 0:
    dist = dist.round(round_x)


# grid = pd.DataFrame(trails, columns=["x", "y"])
# grid_pt = grid[(grid.x > -x) & (grid.x < x)].pivot_table(
#     index="y", columns="x", aggfunc=len, fill_value=0
# )


# if normalize:
#     # Normalizes entire grid
#     grid_pt = grid_pt.div(grid_pt.max().max())

# grid.dropna(inplace=True)
params = dict(
    type_trig=type_trig,
    G=G,
    mass=mass,
    wavelength=wavelength,
    power=power,
    inner_power=inner_power,
    moving_wave=moving_wave,
    edges_slits=edges_slits,
    barrier_points=barrier_points,
    n_particles=n_particles,
    barrier_y=barrier_y,
    dt=dt,
    v_y=v_y,
    x=x,
    y=y,
    round_x=round_x,
    normalize=normalize,
    multiplier=multiplier,
)
if PLOT_3D:
    print("Creating surface plot")
    plot_surface(dist, o_x, o_y, template, params)

if PLOT_Z_DISTRIBUTION:
    print("Creating heatmap plot")
    plot_heatmap(dist, o_x, o_y, template, params)

# TODO ~~~ this plot
# if not plot_density:
#     fig = px.line(
#         grid_pt[grid_pt.index == 5],
#         x = grid_pt.columns.tolist(),
#         y = grid_pt.values,
#         # y = grid_pt.values,
#         # animation_frame=grid_pt.index.tolist(),
#     )
#     # fig = px.histogram(
#     #     grid,
#     #     x="x",
#     #     animation_frame="y",
#     #     # animation_group="z",
#     #     template=template,
#     #     histnorm="probability",
#     #     nbins=500,
#     #     range_x=[-x//2, x//2],
#     #     range_y=[0, 0.2],
#     #     barmode="group",
#     #     # color_discrete_sequence=["crimson", "blue"],
#     # )
#     fig.update_layout(
#         title=str(params), font=dict(size=8), margin=dict(t=50, b=0, l=4, r=4),
#     )
#     fig.write_html(f"plotly_graphs/histogram_research.html")
#     fig.write_html(f"plotly_graphs/histogram_{datetime.datetime.now()}.html")


# fig = px.histogram(grid[grid.y == 30], x="x", template=template, nbins=500, histnorm="probability")
# fig.update_xaxes(visible=False)
# fig.update_yaxes(visible=False)
# fig.write_html(f"plotly_graphs/histogram1_test_{time.time()}.html")
