package apps

import java.io._
import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.caffe._

import libs._
import loaders._
import preprocessing._

// to run this app, the ImageNet training and validation data must be located on
// S3 at s3://sparknet/ILSVRC2012_training/ and s3://sparknet/ILSVRC2012_val/.
// Performance is best if the uncompressed data can fit in memory. If it cannot
// fit, you can replace persist() with persist(StorageLevel.MEMORY_AND_DISK).
// However, spilling the RDDs to disk can cause training to be much slower.
object ImageNetApp {
  val trainBatchSize = 256
  val testBatchSize = 50
  val channels = 3
  val fullWidth = 256
  val fullHeight = 256
  val croppedWidth = 227
  val croppedHeight = 227
  val fullImShape = Array(channels, fullHeight, fullWidth)
  val fullImSize = fullImShape.product

  val workerStore = new WorkerStore()

  def main(args: Array[String]) {
    val numWorkers = args(0).toInt
    val s3Bucket = args(1)
    val conf = new SparkConf()
      .setAppName("ImageNet")
      .set("spark.driver.maxResultSize", "30G")
      .set("spark.task.maxFailures", "1")
      .setExecutorEnv("LD_LIBRARY_PATH", "/usr/local/cuda-7.5/lib64:/root/javacpp-presets/caffe/cppbuild/linux-x86_64/caffe-master/build/lib") // TODO(rkn): get rid of this

    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val sparkNetHome = sys.env("SPARKNET_HOME")
    val logger = new Logger(sparkNetHome + "/training_log_" + System.currentTimeMillis().toString + ".txt")

    val loader = new ImageNetLoader(s3Bucket)
    logger.log("loading train data")
    var trainRDD = loader.apply(sc, "ILSVRC2012_img_train/train.0000", "train.txt")
    logger.log("loading test data")
    val testRDD = loader.apply(sc, "ILSVRC2012_img_val/val.00", "val.txt")

    // convert to dataframes
    val schema = StructType(StructField("data", ArrayType(FloatType), false) :: StructField("label", IntegerType, false) :: Nil)
    var trainDF = sqlContext.createDataFrame(trainRDD.map{ case (a, b) => Row(a.map(x => x.toByte), b)}, schema)
    var testDF = sqlContext.createDataFrame(testRDD.map{ case (a, b) => Row(a.map(x => x.toFloat), b)}, schema)

    // TODO(rkn): fix this
    // logger.log("computing mean image")
    // val meanImage = ComputeMean.computeMeanFromMinibatches(trainMinibatchRDD, fullImShape, numTrainData.toInt)
    // val meanImageBuffer = meanImage.getBuffer()
    // val broadcastMeanImageBuffer = sc.broadcast(meanImageBuffer)

    logger.log("coalescing") // if you want to shuffle your data, replace coalesce with repartition
    trainDF = trainDF.coalesce(numWorkers)
    testDF = testDF.coalesce(numWorkers)

    val numTrainData = trainDF.count()
    logger.log("numTrainData = " + numTrainData.toString)

    val numTestData = testDF.count()
    logger.log("numTestData = " + numTestData.toString)

    val trainPartitionSizes = trainDF.mapPartitions(iter => Array(iter.size).iterator).persist()
    val testPartitionSizes = testDF.mapPartitions(iter => Array(iter.size).iterator).persist()
    trainPartitionSizes.foreach(size => workerStore.put("trainPartitionSize", size))
    testPartitionSizes.foreach(size => workerStore.put("testPartitionSize", size))
    logger.log("trainPartitionSizes = " + trainPartitionSizes.collect().deep.toString)
    logger.log("testPartitionSizes = " + testPartitionSizes.collect().deep.toString)

    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    // initialize nets on workers
    workers.foreach(_ => {
      val netParam = new NetParameter()
      ReadProtoFromTextFileOrDie(sparkNetHome + "/models/bvlc_reference_caffenet/train_val.prototxt", netParam)
      val solverParam = new SolverParameter()
      ReadSolverParamsFromTextFileOrDie(sparkNetHome + "/models/bvlc_reference_caffenet/solver.prototxt", solverParam)
      solverParam.clear_net()
      solverParam.set_allocated_net_param(netParam)

      // TODO(rkn): use mean preprocessor and random cropping!!!
      // Caffe.set_mode(Caffe.GPU)
      val solver = new CaffeSolver(solverParam, schema, new DefaultPreprocessor(schema))
      workerStore.put("netParam", netParam) // prevent netParam from being garbage collected
      workerStore.put("solverParam", solverParam) // prevent solverParam from being garbage collected
      workerStore.put("solver", solver)
    })

    // initialize weights on master
    var netWeights = workers.map(_ => workerStore.get[CaffeSolver]("solver").trainNet.getWeights()).collect()(0)

    var i = 0
    while (true) {
      logger.log("broadcasting weights", i)
      val broadcastWeights = sc.broadcast(netWeights)
      logger.log("setting weights on workers", i)
      workers.foreach(_ => workerStore.get[CaffeSolver]("solver").trainNet.setWeights(broadcastWeights.value))

      if (i % 10 == 0) {
        logger.log("testing", i)
        val testAccuracies = testDF.mapPartitions(
          testIt => {
            val numTestBatches = workerStore.get[Int]("testPartitionSize") / testBatchSize
            var accuracy = 0F
            for (j <- 0 to numTestBatches - 1) {
              val out = workerStore.get[CaffeSolver]("solver").trainNet.forward(testIt)
              accuracy += out("accuracy").get(Array())
            }
            Array[(Float, Int)]((accuracy, numTestBatches)).iterator
          }
        ).cache()
        val accuracies = testAccuracies.map{ case (a, b) => a }.sum
        val numTestBatches = testAccuracies.map{ case (a, b) => b }.sum
        val accuracy = accuracies / numTestBatches
      }

      logger.log("training", i)
      val syncInterval = 50
      trainDF.foreachPartition(
        trainIt => {
          val len = workerStore.get[Int]("trainPartitionSize")
          val startIdx = Random.nextInt(len - syncInterval * trainBatchSize)
          val it = trainIt.drop(startIdx)
          for (j <- 0 to syncInterval - 1) {
            workerStore.get[CaffeSolver]("solver").step(it)
          }
        }
      )

      logger.log("collecting weights", i)
      netWeights = workers.map(_ => { workerStore.get[CaffeSolver]("solver").trainNet.getWeights() }).reduce((a, b) => WeightCollection.add(a, b))
      netWeights.scalarDivide(1F * numWorkers)
      logger.log("weight = " + netWeights.allWeights("conv1")(0).toFlat()(0).toString, i)
      i += 1
    }

    logger.log("finished training")
  }
}
