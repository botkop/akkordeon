package botkop.akkordeon.mkl

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToBatch}
import com.intel.analytics.bigdl.dataset.{ByteRecord, DataSet, LocalDataSet, MiniBatch}

case class DataProvider[T](dl: LocalDataSet[MiniBatch[T]],
                           batchSize: Int,
                           f: (Batch, ActorRef) => Message,
                           name: String)
    extends Stageable {

  lazy val size: Long = dl.size()
  lazy val numBatches: Long = size / batchSize

  override def stage(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new DataProviderActor(this)), name)
}

object DataProvider {
  val b2f: (Batch, ActorRef) => Forward = (b, sentinel) =>
    Forward(sentinel, b.x, b.y)
  val b2v: (Batch, ActorRef) => Validate = (b, sentinel) =>
    Validate(sentinel, b.x, b.y)

  def apply(mode: String, batchSize: Int, name: String): DataProvider[Float] = {
    val trainMean = 0.13066047740239506
    val trainStd = 0.3081078

    val testMean = 0.13251460696903547
    val testStd = 0.31048024

    val trainData = "data/mnist/train-images-idx3-ubyte"
    val trainLabel = "data/mnist/train-labels-idx1-ubyte"
    val validationData = "data/mnist/t10k-images-idx3-ubyte"
    val validationLabel = "data/mnist/t10k-labels-idx1-ubyte"

    mode match {
      case "train" =>
        val trainSet: DataSet[MiniBatch[Float]] = DataSet.array(load(trainData, trainLabel)) ->
          BytesToGreyImg(28, 28) ->
          GreyImgNormalizer(trainMean, trainStd) ->
          GreyImgToBatch(batchSize)
        DataProvider(trainSet.toLocal(), batchSize, b2f, name)
      case "validate" =>
        val validationSet = DataSet.array(load(validationData, validationLabel)) ->
          BytesToGreyImg(28, 28) ->
          GreyImgNormalizer(testMean, testStd) ->
          GreyImgToBatch(batchSize)
        DataProvider(validationSet.toLocal(), batchSize, b2v, name)
    }
  }

  def load(featureFile: String, labelFile: String): Array[ByteRecord] = {
    val featureBuffer =
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    val labelBuffer =
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)))
    val labelMagicNumber = labelBuffer.getInt()

    require(labelMagicNumber == 2049)
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)

    val rowNum = featureBuffer.getInt()
    val colNum = featureBuffer.getInt()

    val result = new Array[ByteRecord](featureCount)
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte](rowNum * colNum)
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = ByteRecord(img, labelBuffer.get().toFloat + 1.0f)
      i += 1
    }
    result
  }

}

class DataProviderActor[T](dp: DataProvider[T]) extends Actor with ActorLogging {

  import dp._

  override def receive: Receive = {
    dl.shuffle()
    val it: Iterator[MiniBatch[T]] = dl.data(false)
    val next: MiniBatch[T] = it.next()
    provide(it, next, 1, 0)
  }

  def provide(di: Iterator[MiniBatch[T]],
              nextBatch: MiniBatch[T],
              epoch: Int,
              startTime: Long): Receive = {
    case NextBatch(r) =>
      r forward f(Batch(nextBatch), sender())
      if (di.hasNext) {
        val t = if (startTime == 0) System.currentTimeMillis() else startTime
        context become provide(di, di.next(), epoch, t)
      } else {
        val duration = System.currentTimeMillis() - startTime
        sender() ! Epoch(name, epoch, duration)
        dl.shuffle()
        val ndi = dl.data(false)
        context become provide(ndi, ndi.next(), epoch + 1, 0)
      }
  }
}
