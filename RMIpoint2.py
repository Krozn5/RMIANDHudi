import pickle
import time
from pyspark.sql.functions import col, collect_list
from pyspark.sql import SparkSession
import argparse

def find_index(target, lst):
    left = 0
    right = len(lst) - 1

    while left <= right:
        mid = (left + right) // 2
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

def getHoodieRecordKey(ind, rmi_model, bt, test_set_x, test_set_y, stage_nums):
    if ind != -1:
        pre1 = rmi_model[0][0].predict(test_set_x[ind])
        if pre1 > stage_nums[1] - 1:
            pre1 = stage_nums[1] - 1
        pre2 = rmi_model[1][pre1].predict(test_set_x[ind])
        if pre2 == test_set_y[ind]:
            return pre2
        else:
            return btree_predict(bt, test_set_x[ind], test_set_y[ind])
    else:
        return -1

def btree_predict(bt, x, y):
    pre = bt.predict(x)
    flag = 1
    pos = pre
    off = 1
    while pos != y:
        pos += flag * off
        flag = -flag
        off += 1
    return pre

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point query using Hudi and RMI model')
    parser.add_argument('--hudi_path', type=str, required=True, help='Path to the Hudi dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the RMI model')
    parser.add_argument('--btree_path', type=str, required=True, help='Path to the BTree model')
    parser.add_argument('--query_value', type=float, required=True, help='Query value for point query')
    parser.add_argument('--feature_column', type=str, required=True, help='Feature column to be used for point query')

    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName("Hudi Example") \
        .config("spark.jars.packages", "org.apache.hudi:hudi-spark3-bundle_2.12:0.9.0") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()

    hudi_df = spark.read \
        .format("org.apache.hudi") \
        .option("path", args.hudi_path) \
        .load()

    df = hudi_df.withColumn("_hoodie_record_key", col("_hoodie_record_key").cast("int"))
    sorted_df = df.orderBy(col("_hoodie_record_key"))
    selected_df = sorted_df.select("_hoodie_record_key", args.feature_column)

    stage_nums = [1, 342]
    train_set_x = selected_df.select(collect_list(args.feature_column)).first()[0]
    train_set_y = selected_df.select(collect_list("_hoodie_record_key")).first()[0]
    test_set_x = train_set_x[:]
    test_set_y = train_set_y[:]

    with open(args.model_path, 'rb') as f:
        rmi_model = pickle.load(f)

    with open(args.btree_path, 'rb') as f:
        bt = pickle.load(f)

    start_time = time.time()

    ind = find_index(args.query_value, test_set_x)
    _hoodie_record_key = getHoodieRecordKey(ind, rmi_model, bt, test_set_x, test_set_y, stage_nums)

    print("Query value's _hoodie_record_key:", _hoodie_record_key)

    df.createOrReplaceTempView("hudi_table")

    sql_query = "SELECT * FROM hudi_table WHERE _hoodie_record_key = {}".format(_hoodie_record_key)
    result_df = spark.sql(sql_query)
    result_df.show()

    end_time = time.time()
    execution_time = end_time - start_time
    count = result_df.count()
    print("Number of records that match the query:", count)
    print("Execution time:", execution_time, "seconds")

    spark.stop()
