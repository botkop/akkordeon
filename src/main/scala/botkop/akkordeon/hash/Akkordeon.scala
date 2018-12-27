package botkop.akkordeon.hash

import akka.actor.ActorSystem
import botkop.{numsca => ns}
import scorch._
import scorch.autograd.Variable
import scorch.nn.{Dropout, Linear, Module}
import scorch.optim._

import scala.language.postfixOps

object Akkordeon extends App {

  implicit val system: ActorSystem = ActorSystem("akkordeon")

  ns.rand.setSeed(231L)

  def makeNet(sizes: List[Int], lrs: List[Double], drops: List[Double]): List[Gate] =
    sizes
      .sliding(2, 1)
      .zipWithIndex
      .map {
        case (l, i) =>
          val m: Module = new Module() {
            val fc = Linear(l.head, l.last)
            val drop = Dropout(drops(i))
            def forward(x: Variable): Variable = x ~> fc ~> drop ~> relu
          }
          val o = DCASGDa(m.parameters, lrs(i))
        Gate(m, o, s"g$i")
    } toList

  def accuracy(yHat: Variable, y: Variable): Double = {
    val guessed = ns.argmax(yHat.data, axis = 1)
    ns.mean(y.data == guessed).squeeze()
  }

  def makeSentinels(takes: List[Int], concurrencies: List[Int]): Unit = {
    for (i <- concurrencies.indices) {
      val tdp = DataProvider("mnist", "train", batchSize, Some(takes(i)), s"tdp$i")
      val ts = Sentinel(tdp, concurrencies(i), softmaxLoss, List(accuracy), s"ts$i").stage
      ts ! Wire(Some(gates.last), Some(gates.head))
      ts ! Start
    }
  }

  val learningRates = List(2e-2, 1e-2, 5e-3)
  val dropOuts = List(0.15, 0.07, 0.03)
  val net = makeNet(List(28 * 28, 50, 20, 10), learningRates, dropOuts)
  // val net = makeNet(List(28 * 28, 50, 20, 10), learningRates, List(.5, .2, .1))
  val gates = Stageable.connect(net)
  val batchSize = 2048

  makeSentinels(List(60000, 30000, 30000), List(1, 1, 1))

  val vdp = DataProvider("mnist", "validate", 1024, None, "vdp")
  val vs1 = Sentinel(vdp, 1, softmaxLoss, List(accuracy), "vs1").stage
  vs1 ! Wire(Some(gates.last), Some(gates.head))

  while (true) {
    Thread.sleep(20000)
    vs1 ! Start
  }

}
