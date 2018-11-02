package botkop.akkordeon.single

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

  def connect(gates: List[Stageable])(
      implicit system: ActorSystem): List[ActorRef] = connect(stage(gates))

}
