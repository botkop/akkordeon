package botkop.akkordeon.flat

import akka.actor.{ActorRef, ActorSystem}
import botkop.akkordeon.Stageable

case class Wire(prev: ActorRef, next: ActorRef)

object Wire {

  def connect(sentinel: ActorRef, gates: List[ActorRef]): List[ActorRef] = {
    gates.zipWithIndex.foreach {
      case (g, i) =>
        val prev = if (i > 0) gates(i - 1) else sentinel
        val next = if (i + 1 < gates.length) gates(i + 1) else sentinel
        g ! Wire(prev, next)
    }
    sentinel ! Wire(gates.last, gates.head)
    sentinel :: gates
  }

  def connect(sentinel: Sentinel, gates: List[Gate])(
      implicit system: ActorSystem): List[ActorRef] = {
    val s = sentinel.stage
    val gs = gates.map(g => g.stage)
    connect(s, gs)
  }

  def connect(sentinel: Sentinel, gates: Gate*)(
      implicit system: ActorSystem): List[ActorRef] =
    connect(sentinel, gates.toList)

  def connect(stageables: List[Stageable])(
      implicit system: ActorSystem): List[ActorRef] = {
    val ss = Stageable.stage(stageables)
    val sentinel = ss.head
    val gates = ss.tail
    connect(sentinel, gates)
  }

}
