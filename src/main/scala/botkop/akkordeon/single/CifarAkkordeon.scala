package botkop.akkordeon.single

import akka.actor.{ActorRef, ActorSystem}
import botkop.{numsca => ns}
import scorch._
import scorch.autograd.Variable
import scorch.nn.cnn.{Conv2d, MaxPool2d}
import scorch.nn.{Linear, Module}
import scorch.optim.DCASGDa

import scala.language.postfixOps

object CifarAkkordeon extends App {

  implicit val system: ActorSystem = ActorSystem("akkordeon")

  botkop.numsca.rand.setSeed(232L)

  val lr = 0.003

  val batchSize = 100

  val net: List[Gate] = makeNet()
  val gates: List[ActorRef] = Stageable.connect(net)

//  val tdp1 = DataProvider("cifar-10", "train", batchSize, None, s"tdp1")
//  val ts1: ActorRef =
//    Sentinel(tdp1, 4, softmaxLoss, List(accuracy), s"ts1").stage
//  ts1 ! Wire(Some(gates.last), Some(gates.head))
//  ts1 ! Start

  val tdp2 = DataProvider("cifar-10", "train", batchSize, None, s"tdp2")
  val ts2: ActorRef =
    Sentinel(tdp2, 2, softmaxLoss, List(accuracy), s"ts2").stage
  ts2 ! Wire(Some(gates.last), Some(gates.head))
  ts2 ! Start

//  val vdp = DataProvider("cifar-10", "dev", batchSize, None, "vdp")
//  val vs: ActorRef = Sentinel(vdp, 1, softmaxLoss, List(accuracy), "vs").stage
//  vs ! Wire(Some(gates.last), Some(gates.head))
//
//  while (true) {
//    Thread.sleep(120000)
//    vs ! Start
//  }

  def makeNet(): List[Gate] = {

    val m1 = new Module() {
      val c1 = Conv2d(numChannels = 3,
                      numFilters = 6,
                      filterSize = 3,
                      weightScale = 1e-3,
                      pad = 1,
                      stride = 1)
      val p2 = MaxPool2d(poolSize = 2, stride = 2)
      override def forward(x: Variable): Variable = {
        x ~> c1 ~> p2
      }
    }

    val m2 = new Module() {
      val c3 = Conv2d(numChannels = 6,
                      numFilters = 16,
                      filterSize = 5,
                      weightScale = 1e-3,
                      pad = 1,
                      stride = 1)
      val p4 = MaxPool2d(poolSize = 2, stride = 2)
      override def forward(x: Variable): Variable = x ~> c3 ~> p4
    }

    val m3 = new Module() {
      val c5 = Conv2d(numChannels = 16,
                      numFilters = 120,
                      filterSize = 5,
                      weightScale = 1e-3,
                      stride = 1,
                      pad = 1)

      val numFlatFeatures: Int = 3000
      def flatten(v: Variable): Variable = v.reshape(-1, numFlatFeatures)

      override def forward(x: Variable): Variable = x ~> c5 ~> flatten
    }

    import scala.language.reflectiveCalls
    val m4 = Linear(m3.numFlatFeatures, 84)
    val m5 = Linear(84, 10)

    val modules = List(m1, m2, m3, m4, m5)

    modules.zipWithIndex.map {
      case (m, i) =>
        val o = DCASGDa(m.parameters, lr)
        Gate(m, o, s"g$i")
    }
  }

  def accuracy(yHat: Variable, y: Variable): Double = {
    val guessed = ns.argmax(yHat.data, axis = 1)
    ns.mean(y.data == guessed).squeeze()
  }

}
