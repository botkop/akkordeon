package botkop.akkordeon.wheels

import akka.actor.ActorSystem
import botkop.akkordeon.Stageable
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

  val lr = 0.0203
  val net = makeNet(lr, 28 * 28, 50, 20, 10)

  val batchSize = 128
  val tdl: DataLoader = DataLoader.instance("mnist", "train", batchSize)
  val vdl: DataLoader = DataLoader.instance("mnist", "dev", batchSize)

  val s = Sentinel(tdl, vdl, 32, 1, softmaxLoss, accuracy, "sentinel")
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

}
