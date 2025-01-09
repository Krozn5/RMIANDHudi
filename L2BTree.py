import numpy as np
import pandas as pd
import gc
import time
from Trained_NN import TrainedNN, AbstractNN, ParameterPool, set_data_type
from btree import BTree
import os
from pyspark.sql.functions import col, collect_list
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import pickle
import argparse

# Setting
BLOCK_SIZE = 1
TOTAL_NUMBER = 350000


def hybrid_training(threshold, use_threshold, stage_nums, core_nums, train_step_nums, batch_size_nums,
                    learning_rate_nums, keep_ratio_nums, train_data_x, train_data_y, test_data_x, test_data_y):
    stage_length = len(stage_nums)
    col_num = stage_nums[1]
    tmp_inputs = [[[] for _ in range(col_num)] for _ in range(stage_length)]
    tmp_labels = [[[] for _ in range(col_num)] for _ in range(stage_length)]
    index = [[None for _ in range(col_num)] for _ in range(stage_length)]
    tmp_inputs[0][0] = train_data_x
    tmp_labels[0][0] = train_data_y
    test_inputs = test_data_x

    for i in range(stage_length):
        for j in range(stage_nums[i]):
            if len(tmp_labels[i][j]) == 0:
                continue
            inputs = tmp_inputs[i][j]
            labels = []
            test_labels = []
            if i == 0:
                divisor = stage_nums[i + 1] * 1.04 / (TOTAL_NUMBER / BLOCK_SIZE)
                labels = [int(k * divisor) for k in tmp_labels[i][j]]
                test_labels = [int(k * divisor) for k in test_data_y]
            else:
                labels = tmp_labels[i][j]
                test_labels = test_data_y

            tmp_index = TrainedNN(threshold[i], use_threshold[i], core_nums[i], train_step_nums[i], batch_size_nums[i],
                                  learning_rate_nums[i], keep_ratio_nums[i], inputs, labels, test_inputs, test_labels)
            tmp_index.train()
            index[i][j] = AbstractNN(tmp_index.get_weights(), tmp_index.get_bias(), core_nums[i], tmp_index.cal_err())
            del tmp_index
            gc.collect()

            if i < stage_length - 1:
                for ind in range(len(tmp_inputs[i][j])):
                    p = index[i][j].predict(tmp_inputs[i][j][ind])
                    if p > stage_nums[i + 1] - 1:
                        p = stage_nums[i + 1] - 1
                    tmp_inputs[i + 1][p].append(tmp_inputs[i][j][ind])
                    tmp_labels[i + 1][p].append(tmp_labels[i][j][ind])

    for i in range(stage_nums[stage_length - 1]):
        if index[stage_length - 1][i] is None:
            continue
        mean_abs_err = index[stage_length - 1][i].mean_err
        if mean_abs_err > threshold[stage_length - 1]:
            index[stage_length - 1][i] = BTree(2)
            index[stage_length - 1][i].build(tmp_inputs[stage_length - 1][i], tmp_labels[stage_length - 1][i])

    return index


def train_index(threshold, use_threshold, train_data, model_name):
    train_set_x = train_data.select("patientunitstayid").repartition(1).rdd.flatMap(lambda x: x).collect()
    train_set_y = train_data.select("_hoodie_record_key").repartition(1).rdd.flatMap(lambda x: x).collect()
    test_set_x = train_set_x[:]
    test_set_y = train_set_y[:]

    start_time = time.time()
    trained_index = hybrid_training(threshold, use_threshold, stage_nums, core_nums, train_step_nums, batch_size_nums,
                                    learning_rate_nums, keep_ratio_nums, train_set_x, train_set_y, [], [])
    end_time = time.time()
    learn_time = end_time - start_time
    print("Build Learned NN time ", learn_time)

    with open(model_name + '.pkl', 'wb') as f:
        pickle.dump(trained_index, f)

    err = 0
    start_time = time.time()
    for ind in range(len(test_set_x)):
        pre1 = trained_index[0][0].predict(test_set_x[ind])
        if pre1 > stage_nums[1] - 1:
            pre1 = stage_nums[1] - 1
        pre2 = trained_index[1][pre1].predict(test_set_x[ind])
        err += abs(pre2 - test_set_y[ind])
    end_time = time.time()
    search_time = (end_time - start_time) / len(test_set_x)
    print("Search time %f " % search_time)
    mean_error = err * 1.0 / len(test_set_x)
    print("mean error = ", mean_error)
    print("*************end Learned NN************\n\n")

    del trained_index
    gc.collect()

    print("*************start BTree************")
    bt = BTree(2)
    start_time = time.time()
    bt.build(test_set_x, test_set_y)
    end_time = time.time()
    build_time = end_time - start_time
    print("Build BTree time ", build_time)

    with open('bt_' + model_name + '.pkl', 'wb') as f:
        pickle.dump(bt, f)

    err = 0
    start_time = time.time()
    for ind in range(len(test_set_x)):
        pre = bt.predict(test_set_x[ind])
        err += abs(pre - test_set_y[ind])
        if err != 0:
            flag = 1
            pos = pre
            off = 1
            while pos != test_set_y[ind]:
                pos += flag * off
                flag = -flag
                off += 1
    end_time = time.time()
    search_time = (end_time - start_time) / len(test_set_x)
    print("Search time ", search_time)
    mean_error = err * 1.0 / len(test_set_x)
    print("mean error = ", mean_error)
    print("*************end BTree************")

    del bt
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train hybrid model using Hudi data.')
    parser.add_argument('--hudi_path', type=str, required=True, help='Path to the Hudi dataset')
    parser.add_argument('--feature_col', type=str, required=True, help='Feature column name')
    parser.add_argument('--label_col', type=str, required=True, help='Label column name')
    parser.add_argument('--model_name', type=str, required=True, help='Name to save the trained model')

    args = parser.parse_args()

    conf = SparkConf()
    conf.set("spark.driver.memory", "6g")
    conf.set("spark.executor.memory", "3g")

    spark = SparkSession.builder \
        .appName("Hudi Example") \
        .config("spark.jars.packages", "org.apache.hudi:hudi-spark3-bundle_2.12:0.9.0") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2047m") \
        .config(conf=conf) \
        .getOrCreate()

    hudi_df = spark.read \
        .format("org.apache.hudi") \
        .option("path", args.hudi_path) \
        .load()

    df = hudi_df.withColumn(args.label_col, col(args.label_col).cast("int"))
    sdf = df.orderBy("row_id")
    sorted_df = sdf.orderBy(col(args.label_col))
    selected_df = sorted_df.select(args.label_col, args.feature_col)
    selected_df.show(44)
    print(selected_df.count())

    threshold = [1, 999]
    use_threshold = [True, False]
    stage_nums = [1, 1200]
    stage_nums[1] = round(selected_df.count() / 10000)
    if stage_nums[1] < 1:
        stage_nums[1] = 1
    core_nums = [[1, 1], [1, 1]]
    train_step_nums = [20000, 20000]
    batch_size_nums = [50, 50]
    learning_rate_nums = [0.0001, 0.0001]
    keep_ratio_nums = [1.0, 1.0]

    train_index(threshold, use_threshold, selected_df, args.model_name)
    spark.stop()
