package botkop.akkordeon.flipflop

import akka.actor.ActorRef

case class Wire(prev: ActorRef, next: ActorRef)
