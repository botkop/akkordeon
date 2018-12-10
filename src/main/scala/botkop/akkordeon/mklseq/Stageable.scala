package botkop.akkordeon.mklseq

import akka.actor.{ActorRef, ActorSystem}

trait Stageable {
  def stage(implicit system: ActorSystem): ActorRef
  def name: String
}

object Stageable {

  def stage(ss: List[Stageable])(implicit system: ActorSystem): List[ActorRef] =
    ss.map(_.stage)

  def connect(gates: List[ActorRef]): List[ActorRef] = {
    (None +: gates.map(g => Some(g)) :+ None).sliding(3, 1).foreach {
      case List(p, c, n) =>
        c.get ! Wire(p, n)
    }
    gates
  }

  def connect(sentinels: List[ActorRef], gates: List[ActorRef]): Unit = {
    connect(gates)
    sentinels.foreach { s =>
      s ! Wire(Some(gates.last), Some(gates.head))
    }
  }

  def connect(gates: List[Stageable])(
      implicit system: ActorSystem): List[ActorRef] = connect(stage(gates))

}
