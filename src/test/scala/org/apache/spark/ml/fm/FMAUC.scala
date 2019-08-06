package org.apache.spark.ml.fm

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.optim.configuration.{Algo, Solver}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}

object FMAUC {
  val spark = SparkSession.builder().appName("FMAUC").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("WARN")

  import spark.implicits._

  def labelData(): Unit = {
    var data: DataFrame = spark.read.parquet("/home/two8g/cvdata/spark-fm/src/test/resources/data").toDF("u_id", "i_id", "label")
    data.printSchema()
    data.select("label").distinct().show()

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

    val u_feature_size = data.select(countDistinct("u_id")).first().getLong(0).toInt
    val i_feature_size = data.select(countDistinct("i_id")).first().getLong(0).toInt
    val feature_size = u_feature_size + i_feature_size
    var df = data.rdd.map((row: Row) =>
      LabeledPoint(row.getAs[Int]("label").toDouble,
        Vectors.sparse(feature_size.toInt,
          Array(row.getAs[Double]("u_id_index").toInt, row.getAs[Double]("i_id_index").toInt + u_feature_size),
          Array(1, 1).map((i: Int) => i.toDouble))
      )).toDF("label", "features")

    df = df.withColumn("label", when($"label" === 0, -1).otherwise(1))
    val pos = df.filter($"label" === 1)
    val neg = df.filter($"label" =!= 1)
    pos.count()
    neg.count()
    df = pos.union(neg.sample(1).limit(pos.count().toInt * 5)).sample(1)
    //
    //
    df.repartition(4).write.mode(SaveMode.Overwrite).parquet("/home/two8g/cvdata/spark-fm/src/test/resources/data_2")
    ////
  }

  def main(args: Array[String]): Unit = {

    val df = spark.read.parquet("/home/two8g/cvdata/spark-fm/src/test/resources/data_2")

    val Array(train, test) = df.sample(1).randomSplit(Array(0.8, 0.2), seed = 1124L)
    val fm = new FactorizationMachines()
      .setAlgo(Algo.fromString("binary classification"))
      .setSolver(Solver.fromString("lbfgs"))
      .setRegParamsL2(0, 0, 0)
      .setDim((1, 1, 8))
      .setStepSize(0.1)
      .setNumPartitions(80)

    val evaluator = new BinaryClassificationEvaluator().setRawPredictionCol("prediction").setMetricName("areaUnderROC")

    val model = fm.fit(train)
    val result = model.transform(test)
    val predictionAndLabel = result.select("prediction", "label")

    predictionAndLabel.show()
    predictionAndLabel.groupBy("prediction", "label").count().show()
    println("areaUnderROC: " + evaluator.evaluate(predictionAndLabel.withColumn("prediction", when($"prediction" === 1, 1.0).otherwise(0.0))))
  }
}
