package botkop.akkordeon.hash

import akka.actor.{ActorRef, ActorSystem}
import botkop.{numsca => ns}
import scorch._
import scorch.autograd.Variable
import scorch.data.loader.DataLoader
import scorch.nn.{Linear, Module}
import scorch.optim._

import scala.language.postfixOps

object Akkordeon extends App {

  implicit val system: ActorSystem = ActorSystem("akkordeon")

  ns.rand.setSeed(232L)

  def makeSentinel(bs: Int, validate: Boolean, name: String): ActorRef = {
    val tdl: DataLoader = DataLoader.instance("mnist", "train", bs)
    val vdl: DataLoader = DataLoader.instance("mnist", "validate", bs)

    val trainingComponents = SentinelComponents(tdl, 32, softmaxLoss)
    val validationComponents =
      if (validate)
        Some(SentinelComponents(vdl, 1, accuracy))
      else None
    Sentinel(trainingComponents, validationComponents, name).stage
  }

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

  val s1 = makeSentinel(batchSize, validate = true, "s1")
  val s2 = makeSentinel(batchSize, validate = false, "s2")
//  val s3 = makeSentinel(batchSize, validate = false, "s3")
//  val s4 = makeSentinel(batchSize, validate = false, "s4")
//  val s5 = makeSentinel(batchSize, validate = false, "s5")
//  val s6 = makeSentinel(batchSize, validate = false, "s6")

  s1 ! Wire(Some(gates.last), Some(gates.head))
  s2 ! Wire(Some(gates.last), Some(gates.head))
//  s3 ! Wire(Some(gates.last), Some(gates.head))
//  s4 ! Wire(Some(gates.last), Some(gates.head))
//  s5 ! Wire(Some(gates.last), Some(gates.head))
//  s6 ! Wire(Some(gates.last), Some(gates.head))

  s1 ! Start
  s2 ! Start
//  s3 ! Start
//  s4 ! Start
//  s5 ! Start
//  s6 ! Start
}
