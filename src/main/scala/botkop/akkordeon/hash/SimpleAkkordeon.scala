package botkop.akkordeon.hash

import akka.actor.ActorSystem
import botkop.{numsca => ns}
import scorch._
import scorch.autograd.Variable
import scorch.nn.{Dropout, Linear, Module}
import scorch.optim.DCASGDa

import scala.language.postfixOps

object SimpleAkkordeon extends App {

  implicit val system: ActorSystem = ActorSystem("akkordeon")

  botkop.numsca.rand.setSeed(232L)

  val imageSize: Int = 28 * 28
  val batchSize = 1024

  val sizes = List(imageSize, 50, 20, 10)
  val learningRates = List(2e-2, 1e-2, 5e-3)
  val dropOuts = List(0.1, 0.1, 0.0001)
  val gates = makeNet(sizes, learningRates, dropOuts)
  val net = Stageable.connect(gates)

  val tdp = DataProvider("mnist", "train", batchSize, None, "tdp")
  val ts = Sentinel(tdp, 5, softmaxLoss, List(accuracy), "ts").stage
  ts ! Wire(Some(net.last), Some(net.head))
  ts ! Start

  val vdp = DataProvider("mnist", "validate", 1024, None, "vdp")
  val vs = Sentinel(vdp, 1, softmaxLoss, List(accuracy), "vs").stage
  vs ! Wire(Some(net.last), Some(net.head))
  while (true) {
    Thread.sleep(20000)
    vs ! Start
  }

  def makeNet(sizes: List[Int],
              learningRates: List[Double],
              dropOuts: List[Double]): List[Gate] =
    sizes
      .sliding(2, 1)
      .zipWithIndex
      .map {
        case (l, i) =>
          val m: Module = new Module() {
            val fc = Linear(l.head, l.last)
            val drop = Dropout(dropOuts(i))
            def forward(x: Variable): Variable = x ~> fc ~> relu ~> drop
          }
          val o = DCASGDa(m.parameters, learningRates(i))
        Gate(m, o, s"g$i")
    } toList

  def accuracy(yHat: Variable, y: Variable): Double = {
    val guessed = ns.argmax(yHat.data, axis = 1)
    ns.mean(y.data == guessed).squeeze()
  }

}
