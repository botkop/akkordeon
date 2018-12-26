package botkop.akkordeon

import akka.actor.ActorRef
import scorch.autograd.Variable

import scala.util.Random

package object hash {

  case object Start

  trait Message
  case class Forward(sentinel: ActorRef, x: Variable, y: Variable, id: String = Random.nextString(4)) extends Message
  case class Backward(sentinel: ActorRef, g: Variable, id: String) extends Message
  case class Validate(sentinel: ActorRef, x: Variable, y: Variable) extends Message

  case class Batch(x: Variable, y: Variable)
  case object Batch {
    def apply(xy: (Variable, Variable)): Batch = Batch(xy._1, xy._2)
  }
  case class NextBatch(recipient: ActorRef)

  case class Epoch(provider: String, n: Int, duration: Long)
}
