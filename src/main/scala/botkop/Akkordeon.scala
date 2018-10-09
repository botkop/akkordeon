package botkop

import akka.actor.ActorSystem
import botkop.Sentinel.Start
import scorch._
import scorch.autograd.Variable
import scorch.data.loader.DataLoader
import scorch.nn._
import scorch.optim.SGD
import botkop.{numsca => ns}

import scala.language.postfixOps

object Akkordeon extends App {

  implicit val system: ActorSystem = ActorSystem("akkordeon")

  val imageSize = 28 * 28
  val batchSize = 16
  val lr = 0.01

  def makeNet(lr: Double, sizes: Int*): List[Gate] = {
    sizes.sliding(2, 1).map { l =>
      val m = new Module() {
        val fc = Linear(l.head, l.last)
        override def forward(x: Variable): Variable = x ~> fc ~> relu
      }
      val o = SGD(m.parameters, lr)
      Gate(m, o)
    } toList
  }

  val net = makeNet(lr, imageSize, 50, 20, 10)

  val tdl = DataLoader.instance("mnist", "train", batchSize)
  val vdl = DataLoader.instance("mnist", "dev", batchSize)

  def accuracy(yHat: Variable, y: Variable): Double = {
    val guessed = ns.argmax(yHat.data, axis = 1)
    ns.mean(y.data == guessed).squeeze()
  }

  val s = Sentinel(tdl, vdl, softmaxLoss, accuracy)
  val ring = Wiring.wire(s, net)

  ring.head ! Start

}

