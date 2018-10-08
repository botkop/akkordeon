package botkop

import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import scorch.autograd.Variable
import scorch.nn.Module
import scorch.optim.Optimizer

case class Gate(m: Module, o: Optimizer) extends Actor {

  import botkop.Gate._

  override def receive: Receive = {
    case wire: Wiring =>
      context become forwardHandle(wire)
  }

  def forwardHandle(wire: Wiring): Receive = {
    case Forward(v) =>
      val result = m(v)
      for (n <- wire.next) n ! Forward(result)
      context become backwardHandle(result, wire)
  }

  def backwardHandle(output: Variable, wire: Wiring): Receive = {
    case Backward(g) =>
      o.zeroGrad()
      output.backward(g)
      o.step()
      for (p <- wire.prev) p ! Backward(g)
      context become forwardHandle(wire)
  }
}

object Gate {

  def stage(g: Gate)(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(g))

  def stage(gates: Gate*)(implicit system: ActorSystem): List[ActorRef] =
    stage(gates: _*)

  def stage(gates: List[Gate])(implicit system: ActorSystem): List[ActorRef] = {
    val actors = gates.map(g => Gate.stage(g))
    wire(actors)
  }

  def wire(gates: List[ActorRef]): List[ActorRef] = {
    gates.zipWithIndex.foreach {
      case (g, i) =>
        val prev = if (i > 0) Some(gates(i - 1)) else None
        val next = if (i + 1 < gates.length) Some(gates(i + 1)) else None
        g ! Wiring(prev, next)
    }
    gates
  }

  case class Forward(v: Variable)
  case class Backward(v: Variable)
  case class Wiring(prev: Option[ActorRef], next: Option[ActorRef])
}
