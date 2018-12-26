package botkop.akkordeon.flipflop

import akka.actor.{ActorRef, ActorSystem}

trait Stageable {
  def stage(implicit system: ActorSystem): ActorRef
  def name: String
}

object Stageable {

  def stage(ss: List[Stageable])(implicit system: ActorSystem): List[ActorRef] =
    ss.map(_.stage)

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

  def connect(stageables: List[Stageable])(
      implicit system: ActorSystem): List[ActorRef] = {
    val ss = Stageable.stage(stageables)
    connect(ss.head, ss.tail)
  }

}
