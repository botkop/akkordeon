package botkop.akkordeon.wheels

import akka.actor.ActorSystem
import botkop.akkordeon.Stageable
import botkop.numsca.Tensor
import botkop.{numsca => ns}
import scorch._
import scorch.autograd.Variable
import scorch.data.loader.DataLoader
import scorch.nn.{Linear, Module}
import scorch.optim._

import scala.language.postfixOps

object Akkordeon extends App {

  implicit val system: ActorSystem = ActorSystem("akkordeon")

  ns.rand.setSeed(231L)

  val lr = 0.01
  val net = makeNet(lr, 28 * 28, 50, 20, 10)

  val batchSize = 128
  val tdl: DataLoader = DataLoader.instance("mnist", "train", batchSize)
  val vdl: DataLoader = DataLoader.instance("mnist", "dev", batchSize)

  val s = Sentinel(tdl, vdl, 4, 1, softmaxLoss, accuracy, "sentinel")
  val ring = Stageable.connect(s :: net)

  ring.head ! Start

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

  def accuracy(yHat: Variable, y: Variable): Double = {
    val guessed = ns.argmax(yHat.data, axis = 1)
    ns.mean(y.data == guessed).squeeze()
  }

  case class Nesterov(parameters: Seq[Variable], var lr: Double, beta: Double = 0.9)
    extends Optimizer(parameters) {

    val vs: Seq[Tensor] = parameters.map(p => ns.zeros(p.shape: _*))

    var iteration = 0

    override def step(): Unit = {
      parameters.zip(vs).foreach {
        case (p, v) =>
          val vPrev = v.copy()
          v *= beta
          v -= lr * p.grad.data
          p.data += (-beta * vPrev) + (1 + beta) * v
      }

      iteration += 1
      if (iteration % 5000 == 0) {
        val dlr = lr / 2
        lr -= dlr
        iteration = 0
        println(s"setting lr = $lr")
      }

    }
  }


  case class SGD(parameters: Seq[Variable], lr: Double)
    extends Optimizer(parameters) {
    var iteration = 0
    var rlr: Double = lr
    lazy val resetEvery: Int = 5 * tdl.numBatches
    lazy val anneal: Double = lr / tdl.numSamples
    override def step(): Unit = {
      parameters.foreach { p =>
        p.data -= p.grad.data * rlr
      }
      iteration += 1
      if (iteration % resetEvery == 0) {
        println(rlr)
        rlr = lr
        iteration = 0
        println(s"resetting lr $rlr")
      } else {
        rlr -= anneal
      }
    }
  }

}
