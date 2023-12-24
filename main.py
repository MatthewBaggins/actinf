from typing import NamedTuple

from plotly import express as px, graph_objects as go
import torch as t

from src.envs import make_envs

from src.utils import seed
from src.thermostat import Thermostat


SEED = 400
DEFAULT_N_ENVS = 1000
DEFAULT_N_ROUNDS = 10000


def main() -> None:
    seed(SEED)

    # model = Thermostat()
    # lr = 1e-4
    # optimizer = t.optim.Adam(model.parameters(), lr, maximize=True)

    # train_hist = train(model, optimizer)

    # fig = px.line(
    #     x=list(range(len(train_hist.gains))), y=train_hist.gains
    # ).update_layout(xaxis_title="round", yaxis_title="gain")
    # # fig.show()
    # fig.write_image("training_history_gains_plot.png")

    # test_envs = make_envs(10, temp_mu=22)
    # test_hist = test(model, test_envs, n_rounds=2000)

    # # fig = px.line(x=list(range(len(test_hist.prefs))), y=test_hist.prefs).update_layout(
    # #     xaxis_title="round", yaxis_title="pref"
    # # )
    # fig = go.Figure()
    # x = list(range(len(test_hist.prefs)))
    # mean_temps = test_hist.env_history[:, :, 0].mean(1)
    # fig.add_traces(
    #     data=[go.Scatter(x=x, y=test_hist.prefs), go.Scatter(x=x, y=mean_temps)],
    #     secondary_ys=[0, 1],
    # )
    # fig.write_image("testing_history_prefs_plot.png")


if __name__ == "__main__":
    main()
