import logging
from collections import defaultdict
import os
from typing import List, Dict

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from il_scale.atari.utils.setup_utils import set_seeds

# A logger for this file
log = logging.getLogger(__name__)

FLOPS = [
    "1e13",
    "2e13",
    "5e13",
    "1e14",
    "2e14",
    "5e14",
    "1e15",
    "2e15",
    "5e15",
    "1e16",
    "2e16",
    "5e16",
    "1e17",
    "2e17",
    "5e17",
    "1e18",
]


class IsoFLOPPlotter:
    def __init__(self, game, plot_type: str, plot_cfg):
        self.game = game
        self.plot_type = plot_type
        self.plot_cfg = plot_cfg
        self.result_table = torch.load(
            f"data_objects/eval_{plot_type}/eval_{plot_type}_{game}.tar",
            map_location="cpu",
        )

        i = 1
        for seed in [0, 1, 2, 3, 4, 5]:
            if os.path.exists(
                f"data_objects/eval_{plot_type}/eval_{plot_type}_{game}_{seed}.tar"
            ):
                print(f"Averaging seed {seed} ...")
                extra_seed = torch.load(
                    f"data_objects/eval_{plot_type}/eval_{plot_type}_{game}_{seed}.tar",
                    map_location="cpu",
                )
                for flop in extra_seed:
                    for param in extra_seed[flop]:
                        self.result_table[flop][param] = (
                            self.result_table[flop][param] * i + extra_seed[flop][param]
                        ) / (i + 1)
                i += 1

        self.tick_font_size = "17"
        self.legend_font_size = "14"
        self.title_font_size = "21"
        self.label_font_size = "20"
        self.legend_loc = plot_cfg.legend_loc
        self.ncol = plot_cfg.ncol

        self.yticks = plot_cfg.yticks
        self.ylabels = plot_cfg.ylabels
        self.ylim_min = plot_cfg.ylim_min
        self.ylim_max = plot_cfg.ylim_max
        self.flop_to_min_params = plot_cfg.flop_to_min_params
        self.flop_to_max_params = plot_cfg.flop_to_max_params

        for flop in plot_cfg.ignore_flops:
            del self.result_table[flop]

        # Cut off the parameters
        new_result_table = defaultdict(dict)
        for flop_budget in self.result_table.keys():
            params = sorted(self.result_table[flop_budget].keys())
            if flop_budget in plot_cfg.param_cut_offs:
                params = params[
                    plot_cfg.param_cut_offs[flop_budget][0] : -plot_cfg.param_cut_offs[
                        flop_budget
                    ][1]
                ]
            values = [self.result_table[flop_budget][param] for param in params]
            for param, value in zip(params, values):
                new_result_table[flop_budget][param] = value
        self.result_table = new_result_table

        self.colors = list(
            reversed(
                sns.color_palette(plot_cfg.color_theme, len(self.result_table.keys()))
            )
        )

    def plot(self):
        plt.style.use("seaborn")
        sns.set_style("whitegrid")
        fig, ax = plt.subplots()

        plt.xscale("log")
        plt.yscale("log")

        plt.ylim(self.ylim_min, self.ylim_max)

        for i, flop_budget in enumerate(
            sorted(self.result_table.keys(), key=lambda x: float(x))
        ):
            params = sorted(self.result_table[flop_budget].keys())
            losses = [self.result_table[flop_budget][param] for param in params]
            plt.errorbar(
                params,
                losses,
                marker="o",
                markersize=9.2,
                linewidth=3.8,
                markeredgewidth=1.6,
                markeredgecolor="#F7F7FF",
                label=flop_budget,
                color=self.colors[i],
            )

        self._plot_per_parabola(self.result_table, self.colors)
        if self.plot_type == "return":
            # plot expert score
            plt.axhline(
                y=self.plot_cfg.expert_score,
                color="black",
                linestyle="--",
                label="Expert",
                alpha=1,
                linewidth=2.2,
            )

        plt.yticks(ticks=self.yticks, labels=self.ylabels, fontsize=self.tick_font_size)
        plt.xticks(fontsize=self.tick_font_size)
        ax.minorticks_off()

        plt.legend(
            fontsize=self.legend_font_size,
            frameon=True,
            loc=self.legend_loc,
            ncol=self.ncol,
        )

        # plt.title(self.game, fontsize=self.title_font_size)

        plt.xlabel("Parameters", fontsize=self.label_font_size)
        plt.ylabel(
            "Returns" if self.plot_type != "loss" else "Dev loss",
            fontsize=self.label_font_size,
        )
        plt.savefig(f"figures/{self.game}/iso_flops_{self.plot_type}_vs_params.pdf")
        plt.close()

    def _plot_per_parabola(self, result_table, colors):
        """
        Fit parabola for each flop budget.
        """
        flop_to_data = self._flop_to_data(result_table)

        # Fit all flop budgets
        for i, flop_budget in enumerate(
            sorted(flop_to_data.keys(), key=lambda x: float(x))
        ):
            X = flop_to_data[flop_budget]["X"]
            y = flop_to_data[flop_budget]["y"]

            min_x = min(np.min(np.exp(X[:, -1])), self.flop_to_min_params[flop_budget])
            max_x = max(np.max(np.exp(X[:, -1])), self.flop_to_max_params[flop_budget])

            poly = PolynomialFeatures(2, include_bias=False)
            X = poly.fit_transform(X)
            reg = LinearRegression().fit(X, y)
            print(f"Per-parabola R^2 (flop: {flop_budget}):", reg.score(X, y))

            xs_aug = [
                [float(flop_budget), np.exp(param)]
                for param in np.linspace(np.log(min_x), np.log(max_x))
            ]
            X_aug = np.log(xs_aug)
            X_aug = poly.fit_transform(X_aug)

            y_predicted = reg.predict(X_aug)
            dataset = [
                (np.exp(x[1]), np.exp(y_pred)) for x, y_pred in zip(X_aug, y_predicted)
            ]
            xs = sorted(list(map(lambda x: x[0], dataset)))
            ys = list(map(lambda x: x[1], sorted(dataset, key=lambda x: x[0])))
            plt.plot(xs, ys, "--", linewidth=1.5, color=colors[i])

    def _flop_to_data(self, result_table):
        # Dataset per flop budget
        flop_to_data = defaultdict(dict)
        for flop_budget in result_table.keys():
            xs = []
            ys = []
            for param in result_table[flop_budget]:
                xs.append([float(flop_budget), param])
                ys.append(result_table[flop_budget][param])

            X, y = np.log(xs), np.log(np.array(ys))
            flop_to_data[flop_budget]["X"] = X
            flop_to_data[flop_budget]["y"] = y

        return flop_to_data


class PowerLawPlotter:
    def __init__(self, game, plot_type, plot_cfg):
        self.game = game
        self.plot_type = plot_type
        self.plot_cfg = plot_cfg
        self.result_table = torch.load(
            f"data_objects/eval_{plot_type}/eval_{plot_type}_{game}.tar",
            map_location="cpu",
        )
        self.flop_params_to_samples = torch.load(
            f"data_objects/flop_params_to_samples/flop_params_to_samples_{game}.tar",
            map_location="cpu",
        )

        i = 1
        for seed in [0, 1, 2, 3, 4, 5]:
            if os.path.exists(
                f"data_objects/eval_{plot_type}/eval_{plot_type}_{game}_{seed}.tar"
            ):
                print(f"Averaging seed {seed} ...")
                extra_seed = torch.load(
                    f"data_objects/eval_{plot_type}/eval_{plot_type}_{game}_{seed}.tar",
                    map_location="cpu",
                )
                for flop in extra_seed:
                    for param in extra_seed[flop]:
                        self.result_table[flop][param] = (
                            self.result_table[flop][param] * i + extra_seed[flop][param]
                        ) / (i + 1)
                i += 1

        self.xticks = plot_cfg.xticks
        self.xlabels = plot_cfg.xlabels

        self.yticks = plot_cfg.yticks
        self.ylabel = plot_cfg.ylabel
        self.ylabels = plot_cfg.ylabels
        self.ylim_min = plot_cfg.ylim_min
        self.ylim_max = plot_cfg.ylim_max
        self.legend_loc = plot_cfg.legend_loc

        self.tick_font_size = "17"
        self.legend_font_size = "18"
        self.title_font_size = "21"
        self.label_font_size = "20"

        self.model_ylabels = plot_cfg.model_ylabels
        self.model_yticks = plot_cfg.model_yticks
        self.model_ylim_min = plot_cfg.model_ylim_min
        self.model_ylim_max = plot_cfg.model_ylim_max
        self.model_legend_loc = plot_cfg.model_legend_loc

        self.samples_ylabels = plot_cfg.samples_ylabels
        self.samples_yticks = plot_cfg.samples_yticks
        self.samples_ylim_min = plot_cfg.samples_ylim_min
        self.samples_ylim_max = plot_cfg.samples_ylim_max
        self.samples_legend_loc = plot_cfg.samples_legend_loc

        if self.plot_type != "loss":
            self.corr_yticks = plot_cfg.corr_yticks
            self.corr_xticks = plot_cfg.corr_xticks
            self.corr_xlabels = plot_cfg.corr_xlabels
            self.corr_ylim_min = plot_cfg.corr_ylim_min
            self.corr_ylim_max = plot_cfg.corr_ylim_max
            self.corr_ylabels = plot_cfg.corr_ylabels

        self.color = plot_cfg.color

        for flop in plot_cfg.ignore_flops:
            del self.result_table[flop]

    def plot(self, cross_validate: bool = False):
        # use all data
        xs = sorted(self.result_table.keys(), key=lambda x: float(x))
        if self.plot_type == "loss":
            ys = np.array([np.min(list(self.result_table[x].values())) for x in xs])
        else:
            ys = np.array([np.max(list(self.result_table[x].values())) for x in xs])
        xs = np.array(list(map(lambda x: float(x), xs)))

        # fit log-linear regression
        X, y = np.expand_dims(np.log(xs), 1), np.log(np.array(ys))
        lr = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()

        if cross_validate:
            avg_rmse, beta_0s, beta_1s, num_in_sample, avg_pi = self._cross_validate(xs, ys, max_clip=(self.plot_cfg.expert_score if self.plot_type == "return" else None))
            print(f'Avg. RMSE: {avg_rmse:.3f}')
            print(f'Avg. PI: {avg_pi:.3f}')

        beta_0_ci = lr.conf_int()[0]
        beta_1_ci = lr.conf_int()[1]
        print(f"Beta 0 CI: {beta_0_ci}")
        print(f"Beta 1 CI: {beta_1_ci}")
        print(f"Beta 0: {lr.params[0]}")
        print(f"Beta 1: {lr.params[1]}")

        plt.style.use("seaborn")
        sns.set_style("whitegrid")
        fig, ax = plt.subplots()

        plt.ylim(self.ylim_min, self.ylim_max)

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("FLOPs", fontsize=self.label_font_size)
        plt.ylabel(self.ylabel, fontsize=self.label_font_size)

        plt.scatter(xs, ys, s=80, color=self.color)
        if self.plot_type == "return":
            # plot expert score
            plt.axhline(
                y=self.plot_cfg.expert_score,
                color="black",
                linestyle="--",
                label="Expert",
                alpha=1,
                linewidth=2.2,
            )

        plt.yticks(ticks=self.yticks, labels=self.ylabels, fontsize=self.tick_font_size)
        plt.xticks(ticks=self.xticks, labels=self.xlabels, fontsize=self.tick_font_size)
        ax.minorticks_off()
        # plt.title(self.game, fontsize=self.title_font_size)

        # plot regression line on log plot
        if self.plot_type == "loss":
            label = f"$\log L = {lr.params[0]:.2f} {lr.params[1]:.2f} \cdot \log C$"
        else:
            label = f"$\log R = {lr.params[0]:.2f} + {lr.params[1]:.2f} \cdot \log C$"
        self._plot_log_line(plt, lr, label=label, color=self.color)

        plt.legend(fontsize=self.legend_font_size, frameon=True, loc=self.legend_loc)

        plt.savefig(f"figures/{self.game}/{self.plot_type}_vs_flops_scaling_law.pdf")
        plt.close()

    def plot_model(self, cross_validate: bool = False):
        # use all data
        xs = sorted(self.result_table.keys(), key=lambda x: float(x))
        if self.plot_type == "loss":
            ys = []
            for flop in xs:
                min_loss = 1e9
                min_param = None
                for param in self.result_table[flop]:
                    if self.result_table[flop][param] < min_loss:
                        min_loss = self.result_table[flop][param]
                        min_param = param
                ys.append(min_param)
        else:
            ys = []
            for flop in xs:
                max_return = -1e9
                max_param = None
                for param in self.result_table[flop]:
                    if self.result_table[flop][param] > max_return:
                        max_return = self.result_table[flop][param]
                        max_param = param
                ys.append(max_param)

        ys = np.array(ys)
        xs = np.array(list(map(lambda x: float(x), xs)))

        # fit log-linear regression
        X, y = np.expand_dims(np.log(xs), 1), np.log(np.array(ys))
        lr = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()

        if cross_validate:
            avg_rmse, beta_0s, beta_1s, num_in_sample, avg_pi = self._cross_validate(xs, ys)
            print(f'Avg. RMSE: {avg_rmse:.3f}')
            print(f'Avg. PI: {avg_pi:.3f}')

        beta_0_ci = lr.conf_int()[0]
        beta_1_ci = lr.conf_int()[1]
        print(f"Beta 0 CI: {beta_0_ci}")
        print(f"Beta 1 CI: {beta_1_ci}")
        print(f"Beta 0: {lr.params[0]}")
        print(f"Beta 1: {lr.params[1]}")

        plt.style.use("seaborn")
        sns.set_style("whitegrid")
        fig, ax = plt.subplots()

        plt.ylim(self.model_ylim_min, self.model_ylim_max)

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("FLOPs", fontsize=self.label_font_size)
        plt.ylabel("Parameters", fontsize=self.label_font_size)

        # plt.title(self.game, fontsize=self.title_font_size)

        plt.scatter(xs, ys, s=80, color=self.color)
        plt.yticks(
            ticks=self.model_yticks,
            labels=self.model_ylabels,
            fontsize=self.tick_font_size,
        )
        plt.xticks(ticks=self.xticks, labels=self.xlabels, fontsize=self.tick_font_size)
        ax.minorticks_off()

        # plot regression line on log plot
        label = f"$\log N = {lr.params[0]:.2f} + {lr.params[1]:.2f} \cdot \log C$"
        self._plot_log_line(plt, lr, label=label, color=self.color)

        plt.legend(
            fontsize=self.legend_font_size, frameon=True, loc=self.model_legend_loc
        )

        plt.savefig(
            f"figures/{self.game}/{self.plot_type}_parameters_vs_flops_scaling_law.pdf"
        )
        plt.close()

    def plot_samples(self, cross_validate: bool = False):
        # use all data
        xs = sorted(self.result_table.keys(), key=lambda x: float(x))
        if self.plot_type == "loss":
            ys = []
            for flop in xs:
                min_loss = 1e9
                min_params = None
                for param in self.result_table[flop]:
                    if self.result_table[flop][param] < min_loss:
                        min_loss = self.result_table[flop][param]
                        min_params = param
                ys.append(self.flop_params_to_samples[flop][min_params])
        else:
            ys = []
            for flop in xs:
                max_return = -1e9
                max_params = None
                for param in self.result_table[flop]:
                    if self.result_table[flop][param] > max_return:
                        max_return = self.result_table[flop][param]
                        max_params = param
                ys.append(self.flop_params_to_samples[flop][max_params])

        ys = np.array(ys)
        xs = np.array(list(map(lambda x: float(x), xs)))

        # fit log-linear regression
        X, y = np.expand_dims(np.log(xs), 1), np.log(np.array(ys))
        lr = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()

        if cross_validate:
            avg_rmse, beta_0s, beta_1s, num_in_sample, avg_pi = self._cross_validate(xs, ys)
            print(f'Avg. RMSE: {avg_rmse:.3f}')
            print(f'Avg. PI: {avg_pi:.3f}')

        beta_0_ci = lr.conf_int()[0]
        beta_1_ci = lr.conf_int()[1]
        print(f"Beta 0 CI: {beta_0_ci}")
        print(f"Beta 1 CI: {beta_1_ci}")
        print(f"Beta 0: {lr.params[0]}")
        print(f"Beta 1: {lr.params[1]}")

        plt.style.use("seaborn")
        sns.set_style("whitegrid")
        fig, ax = plt.subplots()

        plt.xscale("log")
        plt.yscale("log")

        # plt.title(self.game, fontsize=self.title_font_size)

        plt.ylim(self.samples_ylim_min, self.samples_ylim_max)

        plt.xlabel("FLOPs", fontsize=self.label_font_size)
        plt.ylabel("Samples", fontsize=self.label_font_size)

        plt.scatter(xs, ys, s=80, color=self.color)
        plt.yticks(
            ticks=self.samples_yticks,
            labels=self.samples_ylabels,
            fontsize=self.tick_font_size,
        )
        plt.xticks(ticks=self.xticks, labels=self.xlabels, fontsize=self.tick_font_size)
        ax.minorticks_off()

        # plot regression line on log plot
        label = f"$\log D = {lr.params[0]:.2f} + {lr.params[1]:.2f} \cdot \log C$"
        self._plot_log_line(plt, lr, label=label, color=self.color)

        plt.legend(
            fontsize=self.legend_font_size, loc=self.samples_legend_loc, frameon=True
        )

        plt.savefig(
            f"figures/{self.game}/{self.plot_type}_samples_vs_flops_scaling_law.pdf"
        )
        plt.close()

    def plot_correlation(self, cross_validate: bool = False):
        if self.plot_type == "loss":
            return

        self.loss_table = torch.load(
            f"data_objects/eval_loss/eval_loss_{self.game}.tar", map_location="cpu"
        )
        self.return_table = torch.load(
            f"data_objects/eval_return/eval_return_{self.game}.tar", map_location="cpu"
        )

        i = 1
        for seed in [0, 1, 2, 3, 4, 5]:
            if os.path.exists(
                f"data_objects/eval_return/eval_return_{self.game}_{seed}.tar"
            ):
                print(f"Averaging seed {seed} ...")
                extra_seed = torch.load(
                    f"data_objects/eval_return/eval_return_{self.game}_{seed}.tar",
                    map_location="cpu",
                )
                for flop in extra_seed:
                    for param in extra_seed[flop]:
                        self.return_table[flop][param] = (
                            self.return_table[flop][param] * i + extra_seed[flop][param]
                        ) / (i + 1)
                i += 1

        for flop in self.plot_cfg.ignore_flops:
            del self.loss_table[flop]
            del self.return_table[flop]

        # use all data
        flops = sorted(self.loss_table.keys(), key=lambda x: float(x))
        xs = []
        ys = []
        for flop in flops:
            min_loss = 1e9
            min_param = None
            for param in self.loss_table[flop]:
                if self.loss_table[flop][param] < min_loss:
                    min_loss = self.loss_table[flop][param]
                    min_param = param
            ys.append(self.return_table[flop][min_param])
            xs.append(1 / min_loss)

        xs = np.array(list(map(lambda x: float(x), xs)))

        # fit log-linear regression
        X, y = np.expand_dims(np.log(xs), 1), np.log(np.array(ys))
        lr = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()

        if cross_validate:
            avg_rmse, beta_0s, beta_1s, num_in_sample, avg_pi = self._cross_validate(xs, ys)
            print(f'Avg. RMSE: {avg_rmse:.3f}')
            print(f'Avg. PI: {avg_pi:.3f}')

        beta_0_ci = lr.conf_int()[0]
        beta_1_ci = lr.conf_int()[1]
        print(f"Beta 0 CI: {beta_0_ci}")
        print(f"Beta 1 CI: {beta_1_ci}")
        print(f"Beta 0: {lr.params[0]}")
        print(f"Beta 1: {lr.params[1]}")

        plt.style.use("seaborn")
        sns.set_style("whitegrid")
        fig, ax = plt.subplots()

        plt.ylim(self.corr_ylim_min, self.corr_ylim_max)

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("1 / Loss", fontsize=self.label_font_size)
        plt.ylabel("Return", fontsize=self.label_font_size)

        # plt.title(self.game, fontsize=self.title_font_size)

        plt.scatter(xs, ys, s=80, color="#7b0d8a")
        plt.yticks(
            ticks=self.corr_yticks,
            labels=self.corr_ylabels,
            fontsize=self.tick_font_size,
        )
        plt.xticks(
            ticks=self.corr_xticks,
            labels=self.corr_xlabels,
            fontsize=self.tick_font_size,
        )
        ax.minorticks_off()

        # plot regression line on log plot
        label = f"$\log R = {lr.params[0]:.2f} - {lr.params[1]:.2f} \cdot \log L$"
        self._plot_log_line(plt, lr, label=label, color="#7b0d8a")

        plt.legend(fontsize=self.legend_font_size, frameon=True, loc=self.legend_loc)

        plt.savefig(f"figures/{self.game}/{self.plot_type}_vs_loss_scaling_law.pdf")
        plt.close()

    def _plot_log_line(self, plt: plt, lr, label=None, color=None):
        """
        Plot a line from slope and intercept, assuming log axes.
        """
        axes = plt.gca()
        start, stop = np.array(axes.get_xlim())
        x_vals = np.linspace(start=start, stop=stop)
        X = sm.add_constant(np.expand_dims(np.log(x_vals), 1))
        pred_results = lr.get_prediction(X)
        y_vals = pred_results.predicted_mean
        y_vals_upper = pred_results.summary_frame()["mean_ci_upper"].to_numpy()
        y_vals_lower = pred_results.summary_frame()["mean_ci_lower"].to_numpy()
        color = sns.color_palette()[0] if not color else color
        plt.plot(x_vals, np.exp(y_vals), "--", color=color, label=label, linewidth=2.2)
        axes.fill_between(
            x_vals, np.exp(y_vals_lower), np.exp(y_vals_upper), alpha=0.2, color=color
        )

    def _cross_validate(self, xs, ys, n_splits=10, max_clip=None):
        n_splits = max(len(xs) - 6, 2)
        # do cross validation
        tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=n_splits, test_size=1)
        rmses = []
        pred_ints = []
        beta_0 = []
        beta_1 = []
        num_in_sample = []
        for i, (train_idx, dev_idx) in enumerate(tscv.split(xs)):
            X_train = xs[train_idx]
            Y_train = ys[train_idx]
            X_dev = xs[dev_idx]
            Y_dev = ys[dev_idx]
            rmse, lr, is_in_pred_interval = self._fit_and_evaluate(X_train, Y_train, X_dev, Y_dev, max_clip)
            rmses.append(rmse)
            pred_ints.append(float(is_in_pred_interval))
            beta_0_ci = lr.conf_int()[0]
            beta_1_ci = lr.conf_int()[1]
            beta_0.append((beta_0_ci[0], lr.params[0], beta_0_ci[1]))
            beta_1.append((beta_1_ci[1], lr.params[1], beta_1_ci[1]))
            num_in_sample.append(len(train_idx))

        return np.mean(rmses), beta_0, beta_1, num_in_sample, np.mean(pred_ints)

    def _fit_and_evaluate(self, train_xs, train_ys, dev_xs, dev_ys, max_clip = None):
        train_xs = list(map(lambda x: float(x), train_xs))
        dev_xs = list(map(lambda x: float(x), dev_xs))

        # fit log-linear regression
        train_X, train_y = np.expand_dims(np.log(train_xs), 1), np.log(np.array(train_ys))
        lr = sm.OLS(train_y, sm.add_constant(train_X, has_constant='add')).fit()    

        # evaluate on dev
        dev_X, dev_y = np.expand_dims(np.log(dev_xs), 1), np.log(np.array(dev_ys))
        pred_results = lr.get_prediction(sm.add_constant(dev_X, has_constant='add'))
        if max_clip is None:
            y_vals = pred_results.predicted_mean
        else:
            y_vals = np.clip(pred_results.predicted_mean, None, np.log(max_clip))

        pi_lower = np.array(pred_results.summary_frame()["obs_ci_lower"])
        pi_upper = np.array(pred_results.summary_frame()["obs_ci_upper"])
        print(f'Prediction Interval: {np.exp(pi_lower).item():,} <= {np.exp(dev_y).item():,} <= {np.exp(pi_upper).item():,}')
        is_in_pred_interval = np.all((np.exp(dev_y) >= np.exp(pi_lower)) and (np.exp(dev_y) <= np.exp(pi_upper)))

        return np.sqrt(np.square(np.exp(dev_y) - np.exp(y_vals)).mean()), lr, is_in_pred_interval

@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    set_seeds(cfg.atari.seed)

    plot_cfg = cfg.exp.loss if cfg.exp.plot_type == "loss" else cfg.exp.returns

    # plotter = IsoFLOPPlotter(cfg.atari.name, cfg.exp.plot_type, plot_cfg.iso_flop)
    # plotter.plot()

    plotter = PowerLawPlotter(cfg.atari.name, cfg.exp.plot_type, plot_cfg.power_law)

    print("***** Plotting LOSS / RETURN vs. FLOPS *****")
    plotter.plot(cross_validate=True)
    print("***** Plotting PARAMETERS vs. FLOPS *****")
    plotter.plot_model(cross_validate=True)
    print("***** Plotting SAMPLES vs. FLOPS *****")
    plotter.plot_samples(cross_validate=True)
    # print("***** Plotting RETURN vs. LOSS *****")
    # plotter.plot_correlation(cross_validate=True)


if __name__ == "__main__":
    main()
