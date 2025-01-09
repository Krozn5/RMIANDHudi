import argparse
import time
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import row_number, unix_timestamp, lit
from pyspark.sql.window import Window

def main(args):
    start_time = time.time()

    spark = SparkSession.builder \
        .master("local") \
        .appName("csvToHudi") \
        .config("spark.sql.parquet.compression.codec", "none") \
        .config("spark.sql.parquet.binaryAsString", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.jars.packages", "org.apache.hudi:hudi-spark3-bundle_2.12:0.9.0") \
        .getOrCreate()

    # 读取CSV文件
    csvDF = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(args.csv_path)

    # 检查所选列是否存在
    columns = csvDF.columns
    if args.column not in columns:
        raise ValueError(f"Column {args.column} not found in CSV file")

    # 选择指定列
    csvDF = csvDF.select(args.column)

    windowSpec = Window.orderBy(args.column)
    resultDF = csvDF.withColumn("row_id", (row_number().over(windowSpec) - 1)) \
        .withColumn("time", lit(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))) \
        .withColumn("timestamp", unix_timestamp("time", "yyyy-MM-dd HH:mm:ss")) \
        .drop("time")

    resultDF.show(22)

    resultDF.write \
        .mode("overwrite") \
        .format("hudi") \
        .option("hoodie.insert.shuffle.parallelism", "2") \
        .option("hoodie.upsert.shuffle.parallelism", "2") \
        .option("hoodie.table.name", "parquetToHudi") \
        .option("hoodie.datasource.write.recordkey.field", args.column) \
        .option("hoodie.datasource.write.precombine.field", "row_id") \
        .option("hoodie.datasource.write.partitionpath.field", "") \
        .save(args.hudi_path)

    spark.stop()

    execution_time = time.time() - start_time
    print("运行时长：{}s".format(execution_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CSV to Hudi format.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--hudi_path', type=str, required=True, help='HDFS path to save the Hudi table')
    parser.add_argument('--column', type=str, required=True, help='Column to select')

    args = parser.parse_args()
    main(args)
