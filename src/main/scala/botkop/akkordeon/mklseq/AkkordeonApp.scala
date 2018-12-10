package botkop.akkordeon.mklseq

import akka.actor.ActorSystem
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.optim.Top1Accuracy
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine

object AkkordeonApp extends App {

  implicit val system: ActorSystem = ActorSystem("akkordeon")
  val lr = 0.05
  val batchSize = 100

  System.setProperty("bigdl.localMode", "true")
  // System.setProperty("bigdl.coreNumber", "4")
  Engine.init

  implicit val ev: TensorNumeric[Float] = NumericFloat

  val m1 = Sequential[Float]()
    .add(Reshape(Array(1, 28, 28)))
    .add(SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5"))
    .add(Tanh())
    .add(SpatialMaxPooling(2, 2, 2, 2))

  val m2 = Sequential[Float]()
    .add(SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5"))
    .add(Tanh())
    .add(SpatialMaxPooling(2, 2, 2, 2))

  val m3 = Sequential[Float]()
    .add(Reshape(Array(12 * 4 * 4)))
    .add(Linear(12 * 4 * 4, 100).setName("fc1"))
    .add(Tanh())

  val m4 = Sequential[Float]()
    .add(Linear(100, 10).setName("fc2"))
    .add(LogSoftMax())

  val gates = List(m1, m2, m3, m4).zipWithIndex.map { case (m, i) =>
//    val o = new SGD[Float](learningRate = lr)
    val pars = m.modules.map(_.parameters()).filter(_ != null)
    val ps = pars.flatMap(_._1).toArray
    val gs = pars.flatMap(_._2).toArray
    val o = DCASGDa((ps, gs), lr)
    Gate(m, o, s"g$i")
  }

  val net = Stageable.connect(gates)

  val criterion = ClassNLLCriterion[Float]()
  val scoreFunctions = List(new Top1Accuracy[Float]())

  val tdp1 = DataProvider("train", batchSize, "tdp1")
  val ts1 = Sentinel(tdp1, 1, criterion, scoreFunctions, s"ts1").stage
  ts1 ! Wire(Some(net.last), Some(net.head))
  ts1 ! Start

//  val tdp2 = DataProvider("train", batchSize, "tdp2")
//  val ts2 = Sentinel(tdp2, 1, criterion, scoreFunctions, s"ts2").stage
//  ts2 ! Wire(Some(net.last), Some(net.head))
//  ts2 ! Start

  val vdp = DataProvider("validate", batchSize, "vdp")
  val vs = Sentinel(vdp, 1, criterion, scoreFunctions, s"vs").stage
  vs ! Wire(Some(net.last), Some(net.head))

  while (true) {
    Thread.sleep(20000)
    vs ! Start
  }
}
