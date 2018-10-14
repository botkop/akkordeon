package botkop.akkordeon.envelope

import akka.actor.ActorRef

case class Wire(prev: Option[ActorRef], next: Option[ActorRef]) {

}

object Wire {

  def connect(gates: List[ActorRef]): List[ActorRef] = {
    gates.zipWithIndex.foreach {
      case (g, i) =>
        val prev = if (i > 0) Some(gates(i - 1)) else None
        val next = if (i + 1 < gates.length) Some(gates(i + 1)) else None
        g ! Wire(prev, next)
    }
    gates
  }

}
