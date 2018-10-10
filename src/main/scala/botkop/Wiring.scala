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

  def wire(sentinel: Sentinel, gates: List[Gate])(
      implicit system: ActorSystem): List[ActorRef] = {
    val s = sentinel.stage
    val gs = gates.zipWithIndex.map { case (g, i) => g.stage }
    wire(s, gs)
  }

  def wire(sentinel: Sentinel, gates: Gate*)(
      implicit system: ActorSystem): List[ActorRef] =
    wire(sentinel, gates.toList)

  def wire(stageables: List[Stageable])(
      implicit system: ActorSystem): List[ActorRef] = {
    val ss = stageables.map(_.stage)
    val sentinel = ss.head
    val gates = ss.tail
    wire(sentinel, gates)
  }

}
