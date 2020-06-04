# NOTE(patrick): Had to init_engine first on my setup, may be removeable
from bigdl.util.common import *
init_engine()

from zoo.common.nncontext import*
sc = init_nncontext()

import os
import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv")

from zoo.automl.common.util import split_input_df
train_df, val_df, test_df = split_input_df(df, val_split_ratio=0.1, test_split_ratio=0.1)

# build time sequence predictor
from zoo.automl.regression.time_sequence_predictor import *

# you need to specify the name of datetime column and target column
# The default names are "datetime" and "value" respectively.
tsp = TimeSequencePredictor(dt_col="datetime", target_col="value", extra_features_col=None)

from zoo import init_spark_on_local
from zoo.ray import RayContext

ray_ctx = RayContext(sc=sc)
ray_ctx.init()

# Configurable SigOptRecipe for optimization

# How many SigOpt observations to create
observation_budget = 50

# How many suggestions to evaluate in parallel
parallel_bandwidth = 1

# The metrics visible to SigOpt. Metrics have the following properties:
#   name - Required. The name of the metric. Should match the key with which the value is inserted in the Ray result
#   objective - Optional. One of maximize/minmize. The default is maximize.
#   strategy - Optional. One of `optimize` or `store`. Stored metrics are tracked but are not targets for optimization.
#   threshold - Optional. A target value to beat for the metric. Providing this will guide the optimization to favor
#               values that beat all provided thresholds
metrics = [
    {
        'name': 'reward_metric',
        'objective': 'maximize',
        'threshold': -0.4,
        'strategy': 'optimize',
    },
    {
        'name': 'time_total_s',
        'objective': 'minimize',
        'threshold': 9,
        'strategy': 'optimize',
    },
]
recipe = SigOptRecipe(num_samples=observation_budget, max_concurrent=parallel_bandwidth, optimized_metrics=metrics)

pipeline = tsp.fit(train_df, validation_df=val_df, metric="mse", recipe=recipe)
print("Training completed.")

pred_df = pipeline.predict(test_df)

mse, smape = pipeline.evaluate(test_df, metrics=["mse", "smape"])
print(mse)
print(smape)
