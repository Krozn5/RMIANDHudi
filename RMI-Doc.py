import pickle
import time
from pyspark.sql.functions import col, collect_list
from pyspark.sql import SparkSession

def find_index(lst, target):
    if target in lst:
        return lst.index(target)
    else:
        return -1



if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Hudi Example") \
        .config("spark.jars.packages", "org.apache.hudi:hudi-spark3-bundle_2.12:0.9.0") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()

    hudi_df = spark.read \
        .format("org.apache.hudi") \
        .option("path", "hdfs://localhost:9000/lake") \
        .load()

    df = hudi_df.withColumn("_hoodie_record_key", col("_hoodie_record_key").cast("int"))
    df = df.withColumn("nameWithoutExt", col("Column1").cast("int"))

    df.createOrReplaceTempView("hudi_table")

    sorted_df = df.orderBy("_hoodie_record_key")

    # # 对数据进行筛选95
    # filtered_df = hudi_df.filter(col("age") > 25)


    #selected_df = sorted_df.select("_hoodie_record_key", "nameWithoutExt")
    selected_df = sorted_df.select("_hoodie_record_key", "Column1")
    selected_df.show(44)


    stage_nums = [1, 0]  # 每个阶段的模型数量列表
    stage_nums[1] = round(selected_df.count() / 10000)
    if stage_nums[1] < 1:
        stage_nums[1] = 1

    train_set_x = selected_df.select(collect_list("Column1")).first()[0]
    train_set_y = selected_df.select(collect_list("_hoodie_record_key")).first()[0]

    train_set_y = [round(float(i), 2) for i in train_set_y]
    test_set_x = []
    test_set_y = []

    test_set_x = train_set_x[:]
    test_set_y = train_set_y[:]

    # 加载RMI模型
    with open('rmi4Ws_model.pkl', 'rb') as f:
        rmi_model = pickle.load(f)

    flag = 0

    num = int(input("请输入一个整数: "))
    # 记录开始时间
    start_time = time.time()
    ind = find_index(test_set_x,num)
    print("Begin")
    # pick model in next stage
    if(ind != -1):
        pre1 = rmi_model[0][0].predict(test_set_x[ind])
        if pre1 > stage_nums[1] - 1:
            pre1 = stage_nums[1] - 1
        # predict position
        pre2 = rmi_model[1][pre1].predict(test_set_x[ind])
        if(pre2 == test_set_y[ind]):
            print("No error!")
            _hoodie_record_key = pre2
        else: #The predict value of model has error
            flag = 1
    else:
        _hoodie_record_key = -1



    sql_query1 = "SELECT * FROM hudi_table WHERE _hoodie_record_key = %d" % _hoodie_record_key
    sql_query2 = "SELECT * FROM hudi_table WHERE id = %d" % test_set_y[ind]
    if(flag == 1):
        result_df = spark.sql(sql_query1)
    else:
        result_df = spark.sql(sql_query2)

    result_df.show()

    end_time = time.time()
    count = result_df.count()
    print("符合查询条件的记录数为：", count)
    # 计算代码执行时间
    execution_time = end_time - start_time
    print("代码执行时间：", execution_time, "秒")

    spark.stop()


