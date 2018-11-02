package botkop.akkordeon.single


import akka.actor.{ActorRef, ActorSystem}
import botkop.{numsca => ns}
import scorch._
import scorch.autograd.Variable
import scorch.nn.{Linear, Module}
import scorch.optim._

import scala.language.postfixOps

object Akkordeon extends App {

  implicit val system: ActorSystem = ActorSystem("akkordeon")

  ns.rand.setSeed(232L)

  /*
  def makeSentinel(bs: Int, validate: Boolean, name: String): ActorRef = {
    val tdl: DataLoader = DataLoader.instance("mnist", "train", bs)
    val vdl: DataLoader = DataLoader.instance("mnist", "dev", bs)

    val trainingComponents = SentinelComponents(tdl, 32, softmaxLoss)
    val validationComponents =
      if (validate)
        Some(SentinelComponents(vdl, 1, accuracy))
      else None
    Sentinel(trainingComponents, validationComponents, name).stage
  }
  */

  def makeNet(lr: Double, sizes: Int*): List[Gate] =
    sizes
      .sliding(2, 1)
      .zipWithIndex
      .map {
        case (l, i) =>
          val m: Module = new Module() {
            val fc = Linear(l.head, l.last)
            def forward(x: Variable): Variable = x ~> fc ~> relu
          }
          val o = SGD(m.parameters, lr)
        Gate(m, o, s"g$i")
    } toList

  def accuracy(yHat: Variable, y: Variable): Variable = {
    val guessed = ns.argmax(yHat.data, axis = 1)
    Variable(ns.mean(y.data == guessed)) // todo: bit heavy
  }

  val lr = 0.02
  val net = makeNet(lr, 28 * 28, 50, 20, 10)
  val gates = Stageable.connect(net)
  val batchSize = 512

  val tdp1 = DataProvider("mnist", "train", batchSize, None, "tdp1")
  val ts1 = Sentinel(tdp1, 1, softmaxLoss, "ts1").stage
  ts1 ! Wire(Some(gates.last), Some(gates.head))
  ts1 ! Start

  val tdp2 = DataProvider("mnist", "train", batchSize, None, "tdp2")
  val ts2 = Sentinel(tdp2, 1, softmaxLoss, "ts2").stage
  ts2 ! Wire(Some(gates.last), Some(gates.head))
  ts2 ! Start

  val vdp = DataProvider("mnist", "dev", batchSize, None, "tdv")
  val vs1 = Sentinel(vdp, 1, softmaxLoss, "vs1").stage
  vs1 ! Wire(Some(gates.last), Some(gates.head))

  while(true) {
    Thread.sleep(20000)
    vs1 ! Start
  }


}
