package botkop

import akka.actor.{ActorRef, ActorSystem}

case class Wiring(prev: ActorRef, next: ActorRef)

object Wiring {

  def wire(sentinel: ActorRef, gates: List[ActorRef]): List[ActorRef] = {
    gates.zipWithIndex.foreach {
      case (g, i) =>
        val prev = if (i > 0) gates(i - 1) else sentinel
        val next = if (i + 1 < gates.length) gates(i + 1) else sentinel
        g ! Wiring(prev, next)
    }
    sentinel ! Wiring(gates.last, gates.head)
    sentinel :: gates
  }

  def stage(sentinel: Sentinel, gates: List[Gate])(
      implicit system: ActorSystem): List[ActorRef] = {
    val s = Sentinel.stage(sentinel)
    val gs = gates.map(g => Gate.stage(g))
    wire(s, gs)
  }

  def stage(sentinel: Sentinel, gates: Gate*)(
      implicit system: ActorSystem): List[ActorRef] =
    stage(sentinel, gates.toList)

}
