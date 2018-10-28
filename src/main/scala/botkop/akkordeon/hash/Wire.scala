package botkop.akkordeon.hash

import akka.actor.ActorRef

case class Wire(prev: Option[ActorRef], next: Option[ActorRef])

