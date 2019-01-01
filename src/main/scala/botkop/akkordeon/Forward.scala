package botkop.akkordeon

import akka.actor.ActorRef
import scorch.autograd.Variable

import scala.util.Random

case class Forward(sentinel: ActorRef,
                   x: Variable,
                   y: Variable,
                   id: Int = Random.nextInt())
    extends Message
