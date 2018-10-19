package botkop.akkordeon

import akka.actor.ActorRef

case class Wire(prev: ActorRef, next: ActorRef)

