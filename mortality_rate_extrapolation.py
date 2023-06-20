"""Extrapolates period mortality rates using a neural network. (C) Averisera Ltd 2017-2020.

NN is trained to extrapolate a sequence of rates r_0, ..., r_{N-1}, returning r_{N+K-1}.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import collections
import itertools
import logging
import os
import sys
import time
from typing import Optional, Sequence, Tuple

# Replacement for tf.contrib.layers which were removed from TF 2.x.
import numpy as np
import pandas as pd
from tf_slim.layers import layers as _layers
import tensorflow.compat.v1 as tf


# Force the code to run on CPU (faster because our data is small).
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Run TF 1.x code in TF 2.x:
tf.disable_v2_behavior()


# Maximum year for extrapolation.
MAX_EXTRAPOLATION_YEAR = 2200
# Maximum year for which we need the gradients of extrapolated rates.
MAX_EXTRAPOLATION_YEAR_GRAD = 2100
# Ages for which we need the gradients of extrapolated rates.
AGES_GRAD = [0, 5, 10, 15, 20, 30, 45, 60, 65, 75, 85, 90, 100]

# Maximum year for historical data.
MAX_YEAR_HIST = 2019

# Version string
VERSION = "3"

# Where to save results
RESULTS_BASE = "ft_results%d_%s" % (MAX_YEAR_HIST, VERSION)

# Where to save TensorFlow checkpoints
CHECKPOINTS_BASE = "ft_checkpoints%d_%s" % (MAX_YEAR_HIST, VERSION)

# How many extrapolations during training
N_TRAIN = 10

# Loss/bias MA window size in steps (for learning rate decay)
MOVING_AVERAGE_WINDOW_SIZE = 100

# How many times repeat the training and extrapolation when applying the model to full dataset
NUM_APPLY_REPS = 1

SINGLE_FORECAST_HORIZON = 1

INCLUDE_FORECAST_HORIZON_FEATURES_IN_NETWORK_INPUTS = True


# All hyperparameters together
Hyperparameters = collections.namedtuple("Hyperparameters",
                                         [
                                             "trans_name",
                                             # What transformation to apply between rate values and X values.
                                             "n_loss",  # How many extrapolations to compute loss
                                             "delta_steps",  # Interval between learning rate adjustments
                                             # Numbers of steps during training (a list)
                                             "nbr_steps_train",
                                             "initial_learning_rate",
                                             "learning_rate_adjustment_factor",
                                             "batch_size",  # Minibatch size
                                             "input_size",  # Input window size
                                             "num_layers",  # Number of layers except for the input one
                                             "hidden_size",  # Size of each hidden layer
                                             "clip_gradient",  # Whether to clip gradient size
                                         ])


def load_data(rates_path: str, features_paths: Sequence[str]) -> Tuple[Sequence[int], Sequence[int], np.ndarray, np.ndarray, Sequence[str]]:
    """Load data from path.
    
    Returns:
        Sequence of years with rates
        Sequence of age groups with rates
        Mortality rates values
        Features values
        Sequence of feature labels
    """
    rates_df = pd.read_csv(rates_path, parse_dates=False, index_col=0)
    ages = rates_df.index.astype(int)
    years = rates_df.columns.astype(int)
    features_dfs = []
    for features_path in features_paths:
        features_dfs.append(load_features(features_path, years))
    features_df = pd.concat(features_dfs, axis=0)
    features_df.to_csv("features.csv")
    return years, ages, rates_df.values, features_df.values, features_df.columns


def load_features(features_path: str, rates_years: Sequence[int]) -> pd.DataFrame:
    """Loads a set of features from a CSV file with years in the 1st column.
    
    Extrapolates data flat forward and backward to cover the period from min(rates_years) to max(max(rates_years), MAX_EXTRAPOLATION_YEAR).
    """
    min_feature_year = min(rates_years)
    max_feature_year = max(max(rates_years), MAX_EXTRAPOLATION_YEAR)
    features = pd.read_csv(features_path, index_col=0)
    min_feature_year_provided = min(features.index)
    max_feature_year_provided = max(features.index)
    if min_feature_year > min_feature_year_provided:
        features = features.loc[min_feature_year:]
    else:
        for yr in range(min_feature_year, min_feature_year_provided):
            features.loc[yr] = features.loc[min_feature_year_provided]
        features = features.sort_index()
    if max_feature_year < max_feature_year_provided:
        features = features.loc[:max_feature_year]
    else:
        for yr in range(max_feature_year_provided + 1, max_feature_year + 1):
            features.loc[yr] = features.loc[max_feature_year_provided]
        features = features.sort_index()
    assert features.index.min() == min_feature_year
    assert features.index.max() == max_feature_year
    assert len(features) == max_feature_year - min_feature_year + 1, f"Gaps or duplicate years in features data in {features_path}!"
    return features


def create_sequence_queue(
        rates: np.ndarray,
        features: np.ndarray,
        sequence_length: int,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        num_epochs: Optional[int] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Creates a sequence queue from data for NA age groups and NY years.

    Args:
        rates: 2D float array with shape NA x NY.
        features: 2D float array with shape NF x D, where D is the feature dimension and NF >= NY. Must start in the same year as rates.
    """
    _, num_years = rates.shape
    if features.shape[0] > num_years:
        logging.warning("Truncating features to match rates inputs in the temporal dimension")
        features = features[:num_years, :]
    assert features.shape[0] == num_years, "Feature and rates shape mismatch!"
    if sequence_length > num_years:
        raise ValueError("Total sequence length too large")
    feature_dim = features.shape[1]
    print(f"Feature dim == {feature_dim}")
    print(f"Sequence length == {sequence_length}")
    max_num_sequences_per_age = num_years - sequence_length + 1
    rate_data = []
    feature_data = []
    num_sequences = 0
    for rates_for_age in rates:  # Iterate over age groups
        for j in range(max_num_sequences_per_age):
            rate_sequence = rates_for_age[j:(j + sequence_length)]
            assert rate_sequence.shape == (sequence_length,), rate_sequence.shape
            # Use a flattened sequence of features, year after year.
            feature_sequence =  np.reshape(features[j:(j + sequence_length), :], (feature_dim * sequence_length,))
            if np.all(np.isfinite(rate_sequence)):
                rate_data.append(rate_sequence)
                feature_data.append(feature_sequence)
                num_sequences += 1
    assert num_sequences
    rate_data = np.array(rate_data)
    feature_data = np.array(feature_data)
    assert len(rate_data.shape) == 2
    assert len(feature_data.shape) == 2, feature_data.shape
    dataset = tf.data.Dataset.from_tensor_slices((rate_data, feature_data))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    if batch_size is None:
        batch_size = num_sequences
    dataset = dataset.batch(batch_size)
    if num_epochs is not None:
        # repeat the data num_epochs times
        dataset = dataset.repeat(num_epochs)
    else:
        # repeat the data indefinitely
        dataset = dataset.repeat()
    rate_sequences, feature_sequences = dataset.make_one_shot_iterator().get_next()
    return rate_sequences, feature_sequences


def build_cell_neural_network(rates, features, num_layers, hidden_layer_size,
                              output_size, rates_to_inputs, scope_name="fcn"):
    """Build fully connected cell neural network graph for given inputs.

    Args:
        rates: Rate tensor, shape = [batch size, time length]
        num_layers: Number of layers
        hidden_layer_size: Size of the hidden (i.e. each except for the last one) layer
        output_size: Size of the last layer
        rates_to_inputs: Converts rates to input layer
    Returns:
        probability tensor with shape (batch size, output_size)
    """
    if num_layers < 1:
        raise ValueError("Need at least 1 layer")
    rates = rates_to_inputs(rates)
    x = tf.concat([rates, features], axis=1)
    logging.debug("x == %s", x)
    bias_initializer = tf.constant_initializer(0.1)
    for layer_idx in range(num_layers):  # loop over network layers
        # x.shape[0] is the batch size, x.shape[1] is the vector length
        is_hidden = layer_idx + 1 < num_layers  # last layer is not "hidden"
        x_size = int(x.shape[1])  # layer input size
        if is_hidden:
            y_size = hidden_layer_size
            activation = tf.nn.relu
            # Glorot initialisation for ReLU
            weights_stdev = np.sqrt(2. / x_size)
        else:
            y_size = output_size
            activation = None
            # Glorot initialisation for linear activation
            weights_stdev = np.sqrt(1. / x_size)
        logging.debug("Weights standard deviation for layer %d: %g",
                      layer_idx, weights_stdev)
        weight_initializer = tf.truncated_normal_initializer(
            stddev=weights_stdev)
        x = _layers.fully_connected(x, y_size, activation_fn=activation,
                                    weights_initializer=weight_initializer,
                                    biases_initializer=bias_initializer,
                                    reuse=tf.AUTO_REUSE, scope="%s_%d" % (scope_name, layer_idx))
        logging.debug("y[%d] == %s", layer_idx, x)
    return tf.sigmoid(x)  # , x


def get_cell_neural_network_builder(*args, **kwargs):
    """Returns a function which creates a cell NN which will be applied recursively.

    Builder should be called as builder(input_sequence).
    """
    return lambda rates, features: build_cell_neural_network(rates, features, *args, **kwargs)


def build_rnn(cell_builder, rate_total_sequence, feature_total_sequence, feature_dim, input_size=None, output_size=None):     
    _, sequence_length = rate_total_sequence.shape    
    if input_size is None:
        input_size = sequence_length
    if output_size is None:
        output_size = sequence_length - input_size
    print(f"Sequence length: {sequence_length}, Input size: {input_size}, Output size: {output_size}")
    outputs = []
    #logits = []
    rate_input_sequence = rate_total_sequence[:, :input_size]
    feature_input_sequence_length = (input_size + (
        SINGLE_FORECAST_HORIZON if INCLUDE_FORECAST_HORIZON_FEATURES_IN_NETWORK_INPUTS else 0)) * feature_dim
    for i in range(output_size):
        feature_idx_start = i * feature_dim
        feature_input_sequence = feature_total_sequence[:, feature_idx_start:(
            feature_input_sequence_length + feature_idx_start)]
        #output, logit = cell_builder(rate_input_sequence, feature_input_sequence)
        output = cell_builder(rate_input_sequence, feature_input_sequence)
        assert output.shape[1] == 1, output
        #assert logit.shape[1] == 1, logit
        outputs.append(output)
        # logits.append(logit)
        rate_input_sequence = tf.concat([rate_input_sequence[:, 1:], output], axis=1)
    assert len(outputs) == output_size
    #assert len(logits) == output_size
    return tf.concat(outputs, axis=1)  # , tf.concat(logits, axis=1)


def split_indices_into_ABC(n, k):
    """Split indices 0, ..., n-1 into sets A containing k-2/k of them and B and C containg 1/k of them each.
    Set A contains index 0.
    """
    assert k > 2
    B = list(range(1, n, k))
    C = list(range(2, n, k))
    not_A = B + C
    A = [i for i in range(n) if i not in not_A]
    return A, B, C


def run(mode, run_idx, country, sex, hyperparams, restore=False, do_gradients=True, save_checkpoints=False):
    """Args:
        mode: "train", "test" or "apply". Data are divided into sets A (60%), B (20%) and C(20%).
        run_idx: Index of the run (for CI calculations).
        country: "uk", "ew", ...
        sex: "male" or "female"
        hyperparams: Hyperparameters tuple
        restore: Whether to load the trained model from disk.
        do_gradients: Whether to save gradients of outputs over inputs.
        save_checkpoints: Whether to save checkpoints with the model.
    """
    rates_to_inputs = get_input_transformation_function(hyperparams.trans_name)
    rates_basename = "%s-%s-mortality-period-qx-%d.csv" % (
        country, sex, MAX_YEAR_HIST)
    rates_path = os.path.join("sources", rates_basename)    
    years, ages, mortality_rates, features, features_labels = load_data(rates_path, [os.path.join("sources", "gender-pay-gap.csv")])
    if hyperparams.trans_name in ["log", "logit"]:
        # Zero rates are not handled well when using those input transformations.
        mortality_rates = np.clip(mortality_rates, 1e-5, None)    
    logging.info("Mortality rates shape: %s", mortality_rates.shape)
    logging.debug("Mortality rates values: %s", mortality_rates)
    logging.debug("Features shape: %s", features.shape)
    logging.debug("Years == %s", years)
    logging.debug("Ages == %s", ages)
    logging.info("Features labels == %s", features_labels)
    num_ages = len(ages)
    num_years = len(years)
    assert num_ages == mortality_rates.shape[0]
    assert num_years == mortality_rates.shape[1]
    assert features.shape[0] == MAX_EXTRAPOLATION_YEAR - min(years) + 1, (features.shape, min(years), MAX_EXTRAPOLATION_YEAR)
    features_dim = features.shape[1]
    max_input_year = max(years)    

    input_size = hyperparams.input_size
    num_layers = hyperparams.num_layers
    hidden_size = hyperparams.hidden_size

    results_dir = os.path.join(RESULTS_BASE, "%s_IS=%d" %
                               (mode, input_size), str(run_idx))
    checkpoints_dir = os.path.join(
        CHECKPOINTS_BASE, "%s_IS=%d" % (mode, input_size), str(run_idx))
    for directory in [results_dir, checkpoints_dir]:
        ensure_dir(os.path.join(".", directory))

    # Reset calculation graph
    tf.reset_default_graph()

    train_target_size = hyperparams.n_loss
    train_sequence_length = hyperparams.input_size + train_target_size
    # We don't have enough data to use longer test sequences.
    test_sequence_length = train_sequence_length

    cell_nn_builder = get_cell_neural_network_builder(
        num_layers, hidden_size, SINGLE_FORECAST_HORIZON, rates_to_inputs, scope_name="mortality_%s_%s" % (country, sex))

    logging.info("Country = %s, Sex = %s", country, sex)
    logging.info("%s", hyperparams)
    logging.info("Saving in directories %s and %s",
                 results_dir, checkpoints_dir)

    do_extrapolation = mode == "apply"

    if do_extrapolation:
        train_indices = list(range(num_ages))
        test_indices = []
    else:
        A, B, C = split_indices_into_ABC(num_ages, 5)
        if mode == "test":
            train_indices = A + B
            test_indices = C
        else:
            assert mode == "train", mode
            train_indices = A
            test_indices = B

    if test_indices:
        test_rate_sequence, test_feature_sequence = create_sequence_queue(
            mortality_rates[test_indices], features, test_sequence_length, batch_size=None, shuffle=False)
        # Test output and target are rates in [0, 1] range
        #test_outputs, test_logits = build_rnn(cell_nn_builder, test_sequence, input_size=input_size)
        test_outputs = build_rnn(
            cell_nn_builder, test_rate_sequence, test_feature_sequence, features_dim, input_size=input_size)
        test_targets = test_rate_sequence[:, input_size:]

    train_rate_sequence, train_feature_sequence = create_sequence_queue(
        mortality_rates[train_indices], features, train_sequence_length, batch_size=16)
    #train_outputs, train_logits = build_rnn(cell_nn_builder, train_rate_sequence, train_feature_sequence, input_size=input_size)
    train_outputs = build_rnn(
        cell_nn_builder, train_rate_sequence, train_feature_sequence, features_dim, input_size=input_size)
    train_targets = train_rate_sequence[:, input_size:]

    if do_extrapolation:
        # Extrapolate last input_size rates from every age group.
        # Do it age-group-by-age-group to speed up calculation of gradients.
        extrap_input_rates = tf.constant(mortality_rates[:, -input_size:])
        num_extrapolated_years = MAX_EXTRAPOLATION_YEAR - max_input_year
        extrap_input_features_data = features[mortality_rates.shape[1]-input_size:, :]  # has dimension num_extrapolated_years x features_dimension
        assert extrap_input_features_data.shape == (num_extrapolated_years + input_size, features_dim), f"{extrap_input_features_data.shape} != {(num_extrapolated_years, features_dim)}"
        total_num_features = np.prod(extrap_input_features_data.shape)
        extrap_input_features_data = np.reshape(extrap_input_features_data, (1, total_num_features))  # All features from the same year are together
        extrap_input_features = tf.constant(extrap_input_features_data)
        extrap_input_features = tf.tile(extrap_input_features, [num_ages, 1])
        
        #extrap_outputs, extrap_logits = build_rnn(cell_nn_builder, extrap_input_rates, extrap_input_features, output_size=num_extrapolated_years)
        print(f"Rates: {extrap_input_rates.shape}")
        print(f"Features: {extrap_input_features.shape}")
        extrap_outputs = build_rnn(
            cell_nn_builder, extrap_input_rates, extrap_input_features, features_dim, output_size=num_extrapolated_years)
        assert extrap_outputs.shape == (
            num_ages, num_extrapolated_years), extrap_outputs
        if do_gradients:
            gradient_mask = tf.placeholder(
                tf.float64, shape=extrap_outputs.shape)
            extrap_gradient_rates = \
                tf.gradients(tf.reduce_sum(gradient_mask * extrap_outputs), extrap_input_rates,
                             stop_gradients=[gradient_mask])[0]
            assert extrap_gradient_rates.shape == (num_ages, input_size), extrap_gradient_rates
            extrap_gradient_features = \
                tf.gradients(tf.reduce_sum(gradient_mask * extrap_outputs), extrap_input_features,
                             stop_gradients=[gradient_mask])[0]
            assert extrap_gradient_features.shape == (num_ages, total_num_features), extrap_gradient_features
            extrap_gradient = tf.concat([extrap_gradient_rates, extrap_gradient_features], axis=1)

    # minimize L2 deviation
    # train_residuals = train_outputs - train_targets
    #train_loss = _mean_kl_divergence_bernoulli(train_targets, train_logits)
    train_residuals = train_outputs - train_targets
    train_loss = tf.reduce_mean(tf.pow(train_residuals, 2))
    train_bias = tf.reduce_mean(train_residuals)
    if not do_extrapolation:
        #test_loss = _mean_kl_divergence_bernoulli(test_targets, test_logits)
        test_residuals = test_outputs - test_targets
        test_loss = tf.reduce_mean(tf.pow(test_residuals, 2))
        test_bias = tf.reduce_mean(test_residuals)

    # learning rate for RMSProp must be small (it scales it up internally)
    # Graph operations to decrease the learning rate if required
    if mode in ("apply", "test"):
        n_steps = 0 if restore else hyperparams.nbr_steps_train
    else:
        n_steps = hyperparams.nbr_steps_train
    if n_steps:
        # Set up training
        log_learning_rate = tf.Variable(np.log(hyperparams.initial_learning_rate), name="log_learning_rate",
                                        trainable=False)
        log_learning_rate_delta = tf.constant(np.log(hyperparams.learning_rate_adjustment_factor),
                                              name="log_learning_rate_delta")
        decrease_learning_rate = tf.assign_add(log_learning_rate, log_learning_rate_delta,
                                               name="decrease_learning_rate")
        learning_rate = tf.exp(log_learning_rate, name="learning_rate")
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        if hyperparams.clip_gradient:
            gradients, variables = zip(*opt.compute_gradients(train_loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            opt_step = opt.apply_gradients(zip(gradients, variables))
        else:
            opt_step = opt.minimize(train_loss)

    step_delta = min(int(n_steps / 10), hyperparams.delta_steps)
    model_filename = os.path.join(checkpoints_dir, "F2_TR%s_NL%d_HS%d_IS%d.ckpt" % (
        hyperparams.trans_name, hyperparams.num_layers, hyperparams.hidden_size, hyperparams.input_size))
    if save_checkpoints or restore:
        saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    # To allow two processes at the same time.
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4

    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        if restore:
            saver.restore(sess, model_filename)
            logging.info("Model restored.")
        else:
            sess.run(init)
        prev_ma_loss = np.inf
        minibatch_losses = []
        minibatch_biases = []
        ma_loss = np.nan
        ma_bias = np.nan
        for i in range(n_steps):
            _, current_loss, current_bias = sess.run(
                [opt_step, train_loss, train_bias])
            if np.isnan(current_loss):
                logging.error(
                    "Loss is NaN, bailing out after %d steps...", i+1)
                break
            nbr_elapsed_steps = i + 1
            minibatch_losses.append(current_loss)
            minibatch_biases.append(current_bias)
            if i == 0:
                logging.info(
                    "Initial minibatch loss == %g and bias == %g", current_loss, current_bias)
            if nbr_elapsed_steps % step_delta == 0:
                ma_loss = np.mean(
                    minibatch_losses[-MOVING_AVERAGE_WINDOW_SIZE:])
                ma_bias = np.mean(
                    minibatch_biases[-MOVING_AVERAGE_WINDOW_SIZE:])
                if ma_loss >= prev_ma_loss:
                    current_learning_rate = sess.run(learning_rate)
                    new_learning_rate = np.exp(
                        sess.run(decrease_learning_rate))
                    logging.info("Decreasing learning rate from %g to %g",
                                 current_learning_rate, new_learning_rate)
                prev_ma_loss = ma_loss
                logging.info(
                    "Training step %d: moving avg. loss == %g, moving avg. bias == %g", i, ma_loss, ma_bias)
                # if mode == "train":
                #    test_loss_value, test_bias_value = sess.run([test_loss, test_bias])
                #    logging.info("Step %d: test loss == %g, test bias == %g", i, test_loss_value, test_bias_value)
                #    if test_loss_value < lowest_test_loss:
                #        lowest_test_loss = test_loss_value
                #        lowest_test_bias = test_bias_value
                #        lowest_test_loss_step = i
        logging.info(
            "Final training moving avg. loss == %g, moving avg. bias == %g", ma_loss, ma_bias)
        if save_checkpoints:
            save_path = saver.save(sess, model_filename)
            logging.info("Model saved in path: %s" % save_path)
        if mode == "apply":
            min_year = min(years)
            extrap_years = list(range(min_year, MAX_EXTRAPOLATION_YEAR + 1))
            # Save results
            df = pd.DataFrame(index=ages, columns=extrap_years, dtype=float)
            df.index.name = "Age"
            for i, year in enumerate(range(min_year, max_input_year + 1)):
                df[year] = mortality_rates[:, i]
            extrap_output_data = sess.run(extrap_outputs)
            logging.debug("Extrapolation outputs: %s", extrap_output_data)
            for i, year in enumerate(range(max_input_year + 1, MAX_EXTRAPOLATION_YEAR + 1)):
                df[year] = extrap_output_data[:, i]
            df.to_csv(os.path.join(results_dir, "predicted-" + rates_basename))
            logging.info("Saved extrapolation results.")
            if do_gradients:
                extrap_input_years = years[-input_size:]
                extrap_output_years = range(
                    max_input_year + 1, MAX_EXTRAPOLATION_YEAR_GRAD + 1)
                extrap_output_years_features = range(
                    max_input_year + 1 - input_size, MAX_EXTRAPOLATION_YEAR + 1)
                for age in AGES_GRAD:
                    age_idx = age - min(ages)  # Assumes ages have no gaps.
                    df = pd.DataFrame(index=extrap_output_years,
                                      columns=list(extrap_input_years) + [f"{label}-{year}" for year, label in itertools.product(extrap_output_years_features, features_labels)])
                    for j, year in enumerate(extrap_output_years):
                        gradient_mask_data = np.zeros(
                            [num_ages, num_extrapolated_years], dtype=float)
                        gradient_mask_data[age_idx, j] = 1.0
                        extrap_gradient_data = sess.run(extrap_gradient, feed_dict={
                                                        gradient_mask: gradient_mask_data})
                        expected_extrap_gradient_data_shape = (num_ages, input_size + total_num_features)
                        assert extrap_gradient_data.shape == expected_extrap_gradient_data_shape, f"Expected shape {expected_extrap_gradient_data_shape} but got {extrap_gradient_data.shape}"
                        df.loc[year] = extrap_gradient_data[age_idx, :]
                    df.to_csv(os.path.join(
                        results_dir, ("gradient-predicted-%d-" % age) + rates_basename))
                logging.info("Saved gradients.")
        else:
            test_loss_value, test_bias_value = sess.run([test_loss, test_bias])
            logging.info("Final test loss: %g, test bias: %g",
                         test_loss_value, test_bias_value)
            return current_loss, current_bias, test_loss_value, test_bias_value
        # else:
        #    logging.info("Lowest test loss %g after %i steps (corresponding test bias %g)", lowest_test_loss,
        #                 lowest_test_loss_step, lowest_test_bias)
        #    return lowest_test_loss, lowest_test_loss_step, lowest_test_bias


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _logit(p):
    return tf.log(p) - tf.log1p(-p)


def get_input_transformation_function(name):
    # Convert rates into NN inputs.
    if name == "log":
        f = tf.log
    elif name == "logit":
        f = _logit
    elif name == "id":
        def f(x): return x
    else:
        raise ValueError("Unknown data transformation: %s" % name)
    return f


if __name__ == "__main__":
    if len(sys.argv) <= 3:
        print(
            "Run as %s <country> <sex> <mode> [first_rep_index] [random_seed]" % sys.argv[0])
        sys.exit()
    country = sys.argv[1]
    sex = sys.argv[2]
    mode = sys.argv[3]

    if len(sys.argv) > 4:
        first_rep_index = int(sys.argv[4])
    else:
        first_rep_index = 0

    nbr_steps_train = 50000

    log_filename = 'extrapolation_with_features_%s_%s_%s_%s_%d.log' % (
        VERSION, country, sex, mode, nbr_steps_train)
    print("Logging to %s" % log_filename)
    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    if len(sys.argv) > 5:
        rand_seed = int(sys.argv[5])
        logging.info("Setting random seed to %d", rand_seed)
        tf.set_random_seed(rand_seed)

    restore = False
    do_gradients = mode == "apply"
    save_checkpoints = mode == "apply"

    #optimal_hyperparams = Hyperparameters(trans_name='exp', n_loss=10, delta_steps=3000, nbr_steps_train=39001, initial_learning_rate=0.001, learning_rate_adjustment_factor=0.9, batch_size=32, input_size=40, num_layers=4, hidden_size=64, clip_gradient=True)
    if country == "ew":
        if sex == "female":
            optimal_hyperparams = Hyperparameters(trans_name='logit', n_loss=10, delta_steps=3000, nbr_steps_train=50000, initial_learning_rate=1e-4,
                                                  learning_rate_adjustment_factor=0.5, batch_size=32, input_size=40, num_layers=5, hidden_size=256, clip_gradient=False)
        else:
            optimal_hyperparams = Hyperparameters(trans_name='logit', n_loss=10, delta_steps=3000, nbr_steps_train=50000, initial_learning_rate=1e-4,
                                                  learning_rate_adjustment_factor=0.5, batch_size=16, input_size=40, num_layers=5, hidden_size=512, clip_gradient=False)
    else:
        optimal_hyperparams = None

    if mode == "train":
        trans_names = ["logit"]
        n_loss_values = [10]
        delta_steps_values = [3000]
        nbr_steps_train_values = [nbr_steps_train]
        initial_learning_rates = [1e-3, 1e-4]
        learning_rate_adjustment_factors = [0.5, 0.9]
        batch_sizes = [16, 32]
        input_sizes = [40]
        num_layers_values = [5, 6, 7]
        hidden_sizes = [128, 256, 512]
        clip_gradient = False

        hyperparams = list(itertools.product(trans_names, n_loss_values, delta_steps_values, nbr_steps_train_values,
                                             initial_learning_rates, learning_rate_adjustment_factors, batch_sizes, input_sizes,
                                             num_layers_values, hidden_sizes, [clip_gradient]))
        logging.info("Checking %d hyperparameter combinations.",
                     len(hyperparams))
        scan_results = []
        start_hp = Hyperparameters(trans_name='logit', n_loss=10, delta_steps=3000, nbr_steps_train=50000, initial_learning_rate=0.0001,
                                   learning_rate_adjustment_factor=0.9, batch_size=32, input_size=40, num_layers=6, hidden_size=512, clip_gradient=False)
        ensure_dir(RESULTS_BASE)
        scan_results_filename = os.path.join(
            RESULTS_BASE, "scan_results_%s_%s_%d.csv" % (country, sex, nbr_steps_train))
        if start_hp is not None:
            logging.warn(
                "!!Skipping all hyperparameter tuples until this one: %s", start_hp)
        for run_idx, hp in enumerate(hyperparams):
            if start_hp is not None:
                if hp == start_hp:
                    # This and subsequent hyperparameter tuples will be worked on
                    start_hp = None
                    logging.info("Starting from hyperparameters: %s", hp)
                else:
                    # Skip this tuple
                    continue
            trans_name, n_loss, delta_steps, nbr_steps_train, initial_learning_rate, learning_rate_adjustment_factor, batch_size, input_size, num_layers, hidden_size, clip_gradient = hp
            hyperparams = Hyperparameters(trans_name=trans_name, n_loss=n_loss, delta_steps=delta_steps,
                                          nbr_steps_train=nbr_steps_train, initial_learning_rate=initial_learning_rate,
                                          learning_rate_adjustment_factor=learning_rate_adjustment_factor,
                                          batch_size=batch_size, input_size=input_size, num_layers=num_layers,
                                          hidden_size=hidden_size, clip_gradient=clip_gradient)
            time0 = time.time()
            test_stats = run(mode, run_idx, country, sex, hyperparams, restore=restore, do_gradients=do_gradients,
                             save_checkpoints=save_checkpoints)
            time1 = time.time()
            delta_time = time1 - time0
            logging.info("Total training time %g seconds, which is %g seconds per step",
                         delta_time, delta_time / nbr_steps_train)
            scan_results.append(hp + test_stats)
            scan_results_df = pd.DataFrame(scan_results,
                                           columns=["trans_name", "n_loss", "delta_steps", "nbr_steps_train",
                                                    "initial_learning_rate", "learning_rate_adjustment_factor",
                                                    "batch_size", "input_size", "num_layers", "hidden_size",
                                                    "clip_gradient", "final_train_loss", "final_train_bias",
                                                    "final_test_loss", "final_test_bias"])
            scan_results_df.to_csv(scan_results_filename)
            logging.info("Saved latest scan results to %s",
                         scan_results_filename)
    elif mode == "test":
        run(mode, 0, country, sex, optimal_hyperparams, restore=restore,
            do_gradients=do_gradients, save_checkpoints=save_checkpoints)
    else:
        for run_idx in range(first_rep_index, NUM_APPLY_REPS):
            run(mode, run_idx, country, sex, optimal_hyperparams, restore=restore, do_gradients=do_gradients,
                save_checkpoints=save_checkpoints)
    logging.info("--> FINISHED <--")
