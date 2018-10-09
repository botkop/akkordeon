package botkop

import akka.actor.ActorSystem
import botkop.Sentinel.Start
import scorch._
import scorch.autograd.Variable
import scorch.data.loader.DataLoader
import scorch.nn._
import scorch.optim.SGD
import botkop.{numsca => ns}

import scala.collection.immutable

object Akkordeon extends App {

  implicit val system: ActorSystem = ActorSystem("akkordeon")

  val imageSize = 32 * 32
  val batchSize = 16
  val lr = 0.1

  val m1: Module = new Module() {
    val fc = Linear(imageSize, 100)
    override def forward(x: Variable): Variable = x ~> fc ~> relu
  }
  val o1 = SGD(m1.parameters, lr)
  val g1 = Gate(m1, o1)

  val m2: Module = new Module() {
    val fc = Linear(100, 10)
    override def forward(x: Variable): Variable = x ~> fc ~> relu
  }
  val o2 = SGD(m2.parameters, lr)
  val g2 = Gate(m2, o2)

  val tdl: DataLoader = new DataLoader {
    override def numBatches: Int = 10
    override def numSamples: Int = batchSize * numBatches

    val data: immutable.IndexedSeq[(Variable, Variable)] = (1 to numBatches) map { _ =>
      val x = ns.randn(batchSize, imageSize)
      val y = ns.randint(10, batchSize, 1)
      (Variable(x), Variable(y))
    }

    override def iterator: Iterator[(Variable, Variable)] = data.iterator
  }

  def accuracy(yHat: Variable, y: Variable): Double = {
    val guessed = ns.argmax(yHat.data, axis = 1)
    ns.mean(y.data == guessed).squeeze()
  }

  val s = Sentinel(tdl, tdl, softmaxLoss, accuracy)
  val ring = Wiring.stage(s, g1, g2)

  ring.foreach(_ ! Start)

}

