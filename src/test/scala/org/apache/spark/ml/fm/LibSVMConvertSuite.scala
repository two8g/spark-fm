package org.apache.spark.ml.fm

import org.apache.spark.ml.feature.{LabeledPoint, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}

object LibSVMConvertSuite {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("LibSVMConvertSuite").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._
    var data: DataFrame = Seq((1, 1, 1), (1, 2, 0), (2, 1, 1)).toDF("u_id", "i_id", "label")
    val u_id_indexer = new StringIndexer()
      .setInputCol("u_id")
      .setOutputCol("u_id_index")
    val i_id_indexer = new StringIndexer()
      .setInputCol("i_id")
      .setOutputCol("i_id_index")

    val uIndexerModel = u_id_indexer.fit(data)
    data = uIndexerModel.transform(data)
    val iIndexerModel = i_id_indexer.fit(data)
    data = iIndexerModel.transform(data)

    data.show()

    val u_feature_size = data.select(countDistinct("u_id")).first().getLong(0).toInt
    val i_feature_size = data.select(countDistinct("i_id")).first().getLong(0).toInt
    val feature_size = u_feature_size + i_feature_size
    val df = data.rdd.map((row: Row) =>
      LabeledPoint(row.getAs[Int]("label").toDouble,
        Vectors.sparse(feature_size.toInt,
          Array(row.getAs[Double]("u_id_index").toInt, row.getAs[Double]("i_id_index").toInt + u_feature_size),
          Array(1, 1).map((i: Int) => i.toDouble))
      )).toDF("label", "features")

    df.show()
    MLUtils.convertVectorColumnsToML(df).repartition(1).write.mode(SaveMode.Overwrite).format("libsvm").save("./libsvm_data")
    spark.read.format("libsvm").load("./libsvm_data").printSchema()
  }
}
