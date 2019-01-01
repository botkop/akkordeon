package botkop.akkordeon

import akka.actor.ActorRef
import scorch.autograd.Variable

case class Backward(sentinel: ActorRef, g: Variable, id: Int) extends Message
