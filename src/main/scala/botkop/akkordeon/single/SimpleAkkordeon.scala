package botkop.akkordeon.single

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


  val lr = 0.01
//  val lr = 0.023
//   val lr = 0.003
//  val lr = 0.0001
  val imageSize: Int = 28 * 28

  val batchSize = 1024

  val net: List[Gate] = makeNet(lr, List(28 * 28, 50, 20, 10))
  val gates: List[ActorRef] = Stageable.connect(net)

  /*
  val tdp = DataProvider("mnist", "train", batchSize, Some(30000), s"tdp")
  val ts: ActorRef = Sentinel(tdp, 2, softmaxLoss, Nil, s"ts").stage
  ts ! Wire(Some(gates.last), Some(gates.head))
  ts ! Start

  val tdp1 = DataProvider("mnist", "train", batchSize, Some(30000), s"tdp1")
  val ts1: ActorRef = Sentinel(tdp1, 2, softmaxLoss, Nil, s"ts1").stage
  ts1 ! Wire(Some(gates.last), Some(gates.head))
  ts1 ! Start
  */

  val tdp2 = DataProvider("mnist", "train", batchSize, None, s"tdp2")
  val ts2: ActorRef = Sentinel(tdp2, 5, softmaxLoss, List(accuracy), s"ts2").stage
  ts2 ! Wire(Some(gates.last), Some(gates.head))
  ts2 ! Start

  val vdp = DataProvider("mnist", "validate", 1024, None, "vdp")
  val vs: ActorRef = Sentinel(vdp, 1, softmaxLoss, List(accuracy), "vs").stage
  vs ! Wire(Some(gates.last), Some(gates.head))

  while (true) {
    Thread.sleep(20000)
    vs ! Start
  }

  def makeNet(lr: Double, sizes: List[Int]): List[Gate] =
    sizes
      .sliding(2, 1)
      .zipWithIndex
      .map {
        case (l, i) =>
          val m: Module = new Module() {
            val fc = Linear(l.head, l.last)
            def forward(x: Variable): Variable = {
              def res(yHat: Variable): Variable = x + yHat
              x ~> fc ~> relu ~> res
            }
          }
          val o = DCASGDa(m.parameters, lr)
          Gate(m, o, s"g$i")
      } toList

  def accuracy(yHat: Variable, y: Variable): Double = {
    val guessed = ns.argmax(yHat.data, axis = 1)
    ns.mean(y.data == guessed).squeeze()
  }
}
