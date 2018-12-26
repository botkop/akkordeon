package botkop.akkordeon.hash

import akka.actor.{ActorRef, ActorSystem}
import botkop.{numsca => ns}
import scorch._
import scorch.autograd.Variable
import scorch.nn.{Linear, Module}
import scorch.optim.DCASGDa

import scala.language.postfixOps

object SimpleAkkordeon extends App {

  implicit val system: ActorSystem = ActorSystem("akkordeon")

  botkop.numsca.rand.setSeed(232L)

  val imageSize: Int = 28 * 28
  val batchSize = 1024

  val net: List[Gate] =
    makeNet(List(2e-2, 1e-2, 5e-3), List(28 * 28, 50, 20, 10))
  val gates: List[ActorRef] = Stageable.connect(net)

  val tdp2 = DataProvider("mnist", "train", batchSize, None, s"tdp2")
  val ts2: ActorRef =
    Sentinel(tdp2, 5, softmaxLoss, List(accuracy), s"ts2").stage
  ts2 ! Wire(Some(gates.last), Some(gates.head))
  ts2 ! Start

  val vdp = DataProvider("mnist", "validate", 1024, None, "vdp")
  val vs: ActorRef = Sentinel(vdp, 1, softmaxLoss, List(accuracy), "vs").stage
  vs ! Wire(Some(gates.last), Some(gates.head))

  while (true) {
    Thread.sleep(20000)
    vs ! Start
  }

  def makeNet(lr: List[Double], sizes: List[Int]): List[Gate] =
    sizes
      .sliding(2, 1)
      .zipWithIndex
      .map {
        case (l, i) =>
          val m: Module = new Module() {
            val fc = Linear(l.head, l.last)
            def forward(x: Variable): Variable = x ~> fc ~> relu
          }
          val o = DCASGDa(m.parameters, lr(i))
        Gate(m, o, s"g$i")
    } toList

  def accuracy(yHat: Variable, y: Variable): Double = {
    val guessed = ns.argmax(yHat.data, axis = 1)
    ns.mean(y.data == guessed).squeeze()
  }

}
