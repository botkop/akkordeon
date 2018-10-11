package botkop.multi

import akka.actor.ActorRef

case class MultiWire(prev: List[ActorRef], next: List[ActorRef]) {}

object MultiWire {

  def wire(sentinels: List[ActorRef],
           gates: List[ActorRef]): (List[ActorRef], List[ActorRef]) = {
    gates.zipWithIndex.foreach {
      case (g, i) =>
        val prev = if (i > 0) List(gates(i - 1)) else sentinels
        val next = if (i + 1 < gates.length) List(gates(i + 1)) else sentinels
        g ! MultiWire(prev, next)
    }

    sentinels.foreach { s =>
      s ! MultiWire(List(gates.last), List(gates.head))
    }

    (sentinels, gates)
  }

}
