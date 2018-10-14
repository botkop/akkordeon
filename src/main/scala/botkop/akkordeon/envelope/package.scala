package botkop.akkordeon

import akka.actor.ActorRef
import scorch.autograd.Variable

package object envelope {
  case object Start
  case class Forward(v: Variable, sentinel: ActorRef)
  case class Backward(v: Variable, sentinel: ActorRef)
  case class Eval(x: Variable, sentinel: ActorRef)
}
