import time
from pyspark.sql.functions import col, collect_list
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Hudi Example") \
        .config("spark.jars.packages", "org.apache.hudi:hudi-spark3-bundle_2.12:0.9.0") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()

    hudi_df = spark.read \
        .format("org.apache.hudi") \
        .option("path", "hdfs://localhost:9000/tpch") \
        .load()

    hudi_df.printSchema()
    hudi_df.show()

    hudi_df.createOrReplaceTempView("hudi_table")

    num = float(input("请输入数值: "))
    # 记录开始时间
    start_time = time.time()

    sql_query1 = "SELECT * FROM hudi_table WHERE  heartrate = %d" % num
    result_df = spark.sql(sql_query1)

    result_df.show()

    end_time = time.time()
    count = result_df.count()
    print("符合查询条件的记录数为：", count)
    # 计算代码执行时间
    execution_time = end_time - start_time
    print("代码执行时间：", execution_time, "秒")


    column_c_values = [row['_hoodie_file_name'] for row in result_df.select('_hoodie_file_name').collect()]
    print(column_c_values)

