import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import datetime
import torch as t


def plot_surface(grid_pt, o_x, o_y, template, params, show=True, z_values=None):
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=grid_pt.columns.values,
            z=grid_pt.values,
            y=grid_pt.index.values,
            showscale=False,
            hoverinfo=None,
            name="diffraction",
            reversescale=True,
            colorscale="Turbo",
            # colorscale=["black", "red"],
            lighting=dict(diffuse=0.0, specular=0.0, fresnel=0.0, roughness=0.0),
            # cmin=0,
            # cmax=200,
            contours_z=dict(
                project_x=False,
                project_y=False,
                project_z=False,
                highlight=True,
                highlightcolor="red",
                highlightwidth=1,
            ),
            contours_y=dict(
                highlightcolor="red",
                project_y=True,
                project_z=False,
                project_x=False,
                highlight=True,
                width=1,
                highlightwidth=1,
                show=False,
            ),
            contours_x=dict(
                highlightcolor="red",
                project_x=True,
                project_y=False,
                project_z=False,
                highlight=False,
                width=16,
                show=False,
                highlightwidth=1,
            ),
        )
    )
    if z_values is not None:
        # Create the x and y coordinates for the z_values surface
        x_coords = t.linspace(o_x.min().item(), o_x.max().item(), z_values.shape[1])
        y_coords = t.linspace(o_y.min().item(), o_y.max().item(), z_values.shape[0])

        fig.add_trace(
            go.Surface(
                x=x_coords.cpu().numpy(),
                y=y_coords.cpu().numpy(),
                z=z_values.cpu().numpy(),
                showscale=False,
                hoverinfo="z",
                name="gravitational_force",
                colorscale="Viridis",
                opacity=0.5,
            )
        )
    # TODO: add edge traces again
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=o_x,
    #         y=o_y,
    #         z=t.ones_like(o_x) * 0,
    #         marker_color="white",
    #         marker_size=5,
    #         mode="markers",
    #         hoverinfo=None,
    #         name="edges",
    #     )
    # )
    fig.update_layout(template=template)
    fig.update_layout(
        title=str(params),
        font=dict(size=8),
        scene=dict(
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            yaxis_range=[0, grid_pt.index.to_numpy().max() * 1.2],
            # zaxis_range=[-10, 200],
        ),
        margin=dict(t=50, b=0, l=0, r=0),
    )
    # fig.update_layout(scene_aspectmode="data")
    fig.update_layout(scene_aspectmode="manual")
    fig.update_layout(scene_aspectratio=dict(x=5, y=3, z=1))
    # set camera to be looking from above, y axis is up and x is horizontal
    fig.update_layout(
        scene_camera=dict(eye=dict(x=0, y=0, z=4), up=dict(x=0, y=1, z=0))
    )

    fig.write_html(f"plotly_graphs/simulator/surface_research.html")
    # fig.write_html(f"plotly_graphs/simulator/surface_{datetime.datetime.now()}.html")
    if show:
        fig.show(renderer="browser")


def plot_heatmap(grid_pt, o_x, o_y, template, params, show=True):
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=grid_pt.columns.values,
            z=grid_pt.values,
            y=grid_pt.index.values,
            reversescale=True,
            # colorscale=["black", "red"],
            colorscale="Turbo",
        )
    )
    # fig.add_trace(
    #     go.Scatter(x=o_x, y=o_y, marker_size=3, marker_color="white", mode="markers")
    # )
    fig.update_layout(template=template)
    fig.update_layout(
        title=str(params),
        font=dict(size=8),
        margin=dict(t=50, b=0, l=0, r=0),
    )
    fig.write_html("plotly_graphs/simulator/heatmap_research.html")
    # fig.write_html(f"plotly_graphs/simulator/heatmap_{datetime.datetime.now()}.html")
    if show:
        fig.show(renderer="browser")
