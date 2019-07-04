package botkop

import botkop.{numsca => ns}
import akka.actor.ActorRef
import scorch.autograd.Variable

import scala.util.Random

package object akkordeon {

  type DataIterator = Iterator[(Variable, Variable)]

  case object Start

  trait Message extends Serializable
  case class Forward(sentinel: ActorRef,
                     x: Variable,
                     y: Variable,
                     id: Int = Random.nextInt())
      extends Message
  case class Backward(sentinel: ActorRef, g: Variable, id: Int) extends Message
  case class Validate(sentinel: ActorRef, x: Variable, y: Variable)
      extends Message

  case class Batch(x: Variable, y: Variable)
  case object Batch {
    def apply(xy: (Variable, Variable)): Batch = Batch(xy._1, xy._2)
  }

  case class FirstBatch(recipient: ActorRef)

  case class NextBatch(recipient: ActorRef)

  case class Epoch(provider: String, n: Int, duration: Long)

  def accuracy(yHat: Variable, y: Variable): Double = {
    val guessed = ns.argmax(yHat.data, axis = 1)
    ns.mean(y.data == guessed).squeeze()
  }

}
