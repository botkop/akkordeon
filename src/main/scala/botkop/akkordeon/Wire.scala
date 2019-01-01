package botkop.akkordeon

import akka.actor.ActorRef

case class Wire(prev: Option[ActorRef], next: Option[ActorRef])

object Wire {
  def apply(prev: ActorRef, next: ActorRef): Wire = Wire(Some(prev), Some(next))
  def apply(actors: List[ActorRef]): Wire = Wire(actors.last, actors.head)
}
