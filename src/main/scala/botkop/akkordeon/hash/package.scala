package botkop.akkordeon

import akka.actor.ActorRef
import scorch.autograd.Variable

package object hash {

  case object Start

  trait Message
  case class Forward(id: String, sentinel: ActorRef, x: Variable, y: Variable) extends Message
  case class Backward(id: String, sentinel: ActorRef, g: Variable) extends Message
  case class Validate(sentinel: ActorRef, x: Variable, y: Variable) extends Message

  case class Batch(x: Variable, y: Variable)
  case object Batch {
    def apply(xy: (Variable, Variable)): Batch = Batch(xy._1, xy._2)
  }
  case class NextBatch(recipient: ActorRef, f: (Batch) => Message)

  case class Epoch(provider: String, n: Int, duration: Long)
}
