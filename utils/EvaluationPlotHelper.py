#  Copyright (c) 2021-2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.
#

import pandas as pd
import seaborn as sns
from .OSHelper import OSHelper
from .PlotHelper import PlotHelper
import os
import numpy as np
import scipy.stats as stats
from typing import List
from MultiProcessingHelper.MultiProcessingHelper import MultiProcessingHelper
from matplotlib.ticker import StrMethodFormatter


class EvaluationPlotHelper:
    @staticmethod
    def plot_group(data: pd.DataFrame,
                   metrics: [str, ...],
                   palette,
                   n_workers,
                   output_dir,
                   level1_case_ids = None,
                   level1_colors = None,
                   level2_case_ids = None,
                   level2_colors = None,
                   methods: [str] or None = None,
                   parts: [str, ...] or None = None,
                   metric_xlim_dict: dict = None,
                   metric_ylim_dict: dict = None,
                   metric_fmt_dict: dict = None,
                   draw_scatter=False):
        if methods is None:
            methods = []
        if parts is None:
            parts = []
        if metric_xlim_dict is None:
            metric_xlim_dict = {}
        if metric_ylim_dict is None:
            metric_ylim_dict = {}
        if metric_fmt_dict is None:
            metric_fmt_dict = {}

        mph = MultiProcessingHelper()

        method_separated_df = data
        if len(methods) > 0:
            condition = [False] * len(method_separated_df)
            for method in methods:
                condition |= method_separated_df["method"] == method
            method_separated_df = method_separated_df.loc[condition]
        else:
            methods = method_separated_df["method"].unique()

        if len(parts) == 0:
            parts = data["part"].unique()

        summary_excel_dir = os.path.join(output_dir, "excel")
        metric_heat_map_dir = os.path.join(output_dir, "metric_heat_map")
        scatter_dir = os.path.join(output_dir, "scatter")
        pvalue_dir = os.path.join(output_dir, "pvalue")
        OSHelper.mkdirs([summary_excel_dir, metric_heat_map_dir, scatter_dir, pvalue_dir])

        EvaluationPlotHelper.draw_part_boxplots(data=method_separated_df,
                                                parts=parts,
                                                metrics=metrics,
                                                metric_ylim_dict=metric_ylim_dict,
                                                level1_case_ids=level1_case_ids,
                                                level1_colors=level1_colors,
                                                level2_case_ids=level2_case_ids,
                                                level2_colors=level2_colors,
                                                palette=palette,
                                                save_dir=os.path.join(output_dir, "boxplot"),
                                                n_workers=n_workers)

        args = []
        for metric in metrics:
            part_mean_matrix = np.zeros((len(methods), len(parts)), dtype=np.float32)
            part_std_matrix = np.zeros_like(part_mean_matrix)
            mean_matrix = np.zeros((len(methods), 1), dtype=np.float32)
            std_matrix = np.zeros_like(mean_matrix)
            for i, method in enumerate(methods):
                method_df = method_separated_df.loc[method_separated_df["method"] == method]
                values = method_df[metric]
                mean_matrix[i] = values.mean()
                std_matrix[i] = values.std()
                for j, part in enumerate(parts):
                    values = method_df.loc[method_df["part"] == part][metric]
                    part_mean_matrix[i][j] = values.mean()
                    part_std_matrix[i][j] = values.std()
            fmt = metric_fmt_dict[metric] if metric in metric_fmt_dict else None

            func_args = [(part_mean_matrix,
                          fmt,
                          parts,
                          methods,
                          metric_heat_map_dir,
                          summary_excel_dir,
                          "method_part_mean_{}".format(metric)),
                         (part_std_matrix,
                          fmt,
                          parts,
                          methods,
                          metric_heat_map_dir,
                          summary_excel_dir,
                          "method_part_std_{}".format(metric)),
                         (mean_matrix,
                          fmt,
                          [metric],
                          methods,
                          metric_heat_map_dir,
                          summary_excel_dir,
                          "method_mean_{}".format(metric)),
                         (std_matrix,
                          fmt,
                          [metric],
                          methods,
                          metric_heat_map_dir,
                          summary_excel_dir,
                          "method_std_{}".format(metric))]
            if n_workers > 0:
                args.extend(func_args)
            else:
                # mph.run(args=func_args, n_workers=0, func=EvaluationPlotHelper._draw_method_heat_map_and_excel)
                for func_arg in func_args:
                    EvaluationPlotHelper._draw_method_heat_map_and_excel(*func_arg)

        if len(args) > 0:
            mph.run(args=args,
                    n_workers=n_workers,
                    func=EvaluationPlotHelper._draw_method_heat_map_and_excel,
                    desc="Draw metric heatmap")

        #
        metric_df_dict = {metric: {part: None for part in parts} for metric in metrics}
        for part in parts:
            for method in methods:
                part_method_df = method_separated_df.loc[(method_separated_df["method"] == method) &
                                                         (method_separated_df["part"] == part)]
                for metric in metrics:
                    temp_df = pd.DataFrame(data={"case_id": part_method_df["case_id"],
                                                 "case_name": part_method_df["case_name"],
                                                 method: part_method_df[metric]})
                    if metric_df_dict[metric][part] is None:
                        metric_df_dict[metric][part] = temp_df
                    else:
                        metric_df_dict[metric][part] = metric_df_dict[metric][part].merge(temp_df,
                                                                                          on=["case_id", "case_name"],
                                                                                          how="inner")

        for metric, part_df_dict in metric_df_dict.items():
            for part, df in part_df_dict.items():
                df = df.loc[:, ~df.columns.str.match("Unnamed")]
                df.sort_values(axis=0, by="case_name", inplace=True)
                df.index = [i for i in range(len(df))]
                df.to_excel(os.path.join(summary_excel_dir, "{}_{}.xlsx".format(part, metric)), engine="openpyxl")

        if draw_scatter:
            p_matrixes = {}
            args = []
            for i, method_x in enumerate(methods):
                for j, method_y in enumerate(methods):
                    if i >= j:
                        continue
                    for part in parts:
                        if part not in p_matrixes:
                            p_matrixes[part] = {}
                        method_x_data = method_separated_df.loc[(method_separated_df["part"] == part) &
                                                                (method_separated_df["method"] == method_x)]
                        method_y_data = method_separated_df.loc[(method_separated_df["part"] == part) &
                                                                (method_separated_df["method"] == method_y)]
                        for metric in metrics:
                            if metric not in p_matrixes[part]:
                                p_matrixes[part][metric] = np.zeros((len(methods), len(methods)), dtype=np.float32)

                            method_x_metric_data = pd.DataFrame(data={method_x: method_x_data[metric],
                                                                      "case_name": method_x_data["case_name"]})

                            method_y_metric_data = pd.DataFrame(data={method_y: method_y_data[metric],
                                                                      "case_name": method_y_data["case_name"]})
                            data = method_x_metric_data.merge(method_y_metric_data, how="inner", on="case_name")
                            statistic, pvalue = stats.wilcoxon(data[method_x], data[method_y])
                            pvalue *= 6
                            ylim = metric_ylim_dict[metric] if metric in metric_ylim_dict else None
                            xlim = metric_xlim_dict[metric] if metric in metric_xlim_dict else None

                            func_arg = {"data": data,
                                        "x": method_x,
                                        "y": method_y,
                                        "xlim": xlim,
                                        "ylim": ylim,
                                        "title": "{} {} (p={:.5f})".format(part, metric, pvalue),
                                        "save_path": os.path.join(scatter_dir,
                                                                  "{}_{}_{}_{}_scatter.png".format(method_x,
                                                                                                   method_y,
                                                                                                   part,
                                                                                                   metric))}
                            if n_workers > 0:
                                args.append((PlotHelper.draw_scatter, func_arg))
                            else:
                                PlotHelper.draw_scatter(**func_arg)
                            p_matrixes[part][metric][i][j] = pvalue

            if len(args) > 0:
                mph.run(args=args, n_workers=n_workers, desc="Draw scatters")

            args = []
            significant_levels = ["0.05", "0.01", "0.001"]
            for part in p_matrixes:
                for metric in p_matrixes[part].keys():
                    p_matrix = p_matrixes[part][metric]

                    func_arg = {"data": p_matrix,
                                "title": "{} {} p-value".format(part, metric),
                                "xticklabels": methods,
                                "yticklabels": methods,
                                "annot": True,
                                "fmt": ".4f",
                                "save_path": os.path.join(pvalue_dir,
                                                          "{}_{}_pvalue.png".format(part, metric)),
                                "half": True,
                                "k": 1}

                    if n_workers > 0:
                        args.append((PlotHelper.draw_heat_map, func_arg))
                    else:
                        PlotHelper.draw_heat_map(**func_arg)

                    for significant_level in significant_levels:
                        func_arg = {"data": p_matrix < float(significant_level),
                                    "title": "{} {} p-value < {}".format(part, metric, significant_level),
                                    "xticklabels": methods,
                                    "yticklabels": methods,
                                    "save_path": os.path.join(pvalue_dir,
                                                              "{}_{}_pvalue_{}.png".format(part,
                                                                                           metric,
                                                                                           significant_level)),
                                    "half": True,
                                    "k": 1}
                        if n_workers > 0:
                            args.append((PlotHelper.draw_heat_map, func_arg))
                        else:
                            PlotHelper.draw_heat_map(**func_arg)

            if len(args) > 0:
                mph.run(args=args, n_workers=n_workers, desc="Draw pvalue heat maps")
                # FunctionHelper.multiprocessing_proxy(args=args, workers=n_workers, desc="Draw pvalue heat maps")

    @staticmethod
    def draw_part_boxplots(data: pd.DataFrame,
                           parts: [str, ...],
                           metrics: [str, ...],
                           palette: sns.color_palette,
                           save_dir: str,
                           level1_case_ids: [int or str, ...] = None,
                           level2_case_ids: {str: [int or str, ...]} = None,
                           level1_colors: [(int, ...), ...] = None,
                           level2_colors: {str: [int, ...]} = None,
                           methods: [str, ...] = None,
                           modes: [str, ...] = None,
                           hue=None,
                           hue_order=None,
                           metric_ylim_dict: dict = None,
                           metric_yunit_dict: dict = None,
                           prefix="",
                           n_workers=0):
        if methods is None:
            methods = []
        if modes is None:
            modes = []
        if metric_ylim_dict is None:
            metric_ylim_dict = {}
        if metric_yunit_dict is None:
            metric_yunit_dict = {}

        mph = MultiProcessingHelper()

        if len(methods) > 0 or len(modes) > 0:
            condition = [False] * len(data)
            for method in methods:
                condition |= data["method"] == method
            # for mode in modes:
            #     condition |= data["mode"] == mode
            data = data.loc[condition]

        OSHelper.mkdirs(save_dir)
        args = []
        print("Draw box plots")
        for part in parts:
            for metric in metrics:
                ylim = metric_ylim_dict[metric] if metric in metric_ylim_dict else None
                yunit = metric_yunit_dict[metric] if metric in metric_yunit_dict else None

                fun_arg = {"data": data.loc[data["part"] == part],
                           "x": "method",
                           "y": metric,
                           "hue": hue,
                           "order": methods if len(methods) > 1 else None,
                           "level1_case_ids": level1_case_ids,
                           "level2_case_ids": level2_case_ids,
                           "level1_colors": level1_colors,
                           "level2_colors": level2_colors,
                           "hue_order": hue_order,
                           "palette": palette,
                           "ylim": ylim,
                           "yunit": yunit,
                           "ticker_format": StrMethodFormatter("{x:04.1f}"),
                           "save_path": os.path.join(save_dir, "{}{}_{}.png".format(prefix, part, metric))}

                if n_workers > 0:
                    args.append((PlotHelper.draw_boxplot, fun_arg))
                else:
                    PlotHelper.draw_boxplot(**fun_arg)
        if len(args) > 0:
            mph.run(args=args, n_workers=n_workers, desc="Draw box plot")

    @staticmethod
    def _draw_method_heat_map_and_excel(data: np.ndarray,
                                        fmt: str,
                                        xticklabels: List[str],
                                        yticklabels: List[str],
                                        images_dir: str,
                                        excels_dir: str,
                                        filename: str):
        OSHelper.mkdirs(images_dir)
        OSHelper.mkdirs(excels_dir)

        PlotHelper.draw_heat_map(data=data,
                                 annot=True,
                                 fmt=fmt,
                                 xticklabels=xticklabels,
                                 yticklabels=yticklabels,
                                 save_path=os.path.join(images_dir, "{}.png".format(filename)))
        data = np.squeeze(data)
        if data.ndim == 1:
            pd.DataFrame(data={"value": data},
                         index=yticklabels
                         ).to_excel(os.path.join(excels_dir,
                                                 "{}.xlsx".format(filename)),
                                    engine="openpyxl")
        else:
            pd.DataFrame(data={yticklabels[i]: data[i, :] for i in range(len(yticklabels))},
                         index=xticklabels
                         ).to_excel(os.path.join(excels_dir,
                                                 "{}.xlsx".format(filename)),
                                    engine="openpyxl")

    pass
