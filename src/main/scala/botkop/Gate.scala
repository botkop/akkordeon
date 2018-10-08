package botkop

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import scorch.autograd.Variable
import scorch.nn.Module
import scorch.optim.Optimizer

case class Gate(module: Module, optimizer: Optimizer)

class GateActor(gate: Gate) extends Actor with ActorLogging {

  import gate._
  import botkop.Gate._

  log.debug("instantiating gate actor")

  override def receive: Receive = {
    case wire: Wiring =>
      log.debug(s"received wire $wire")
      context become forwardHandle(wire)
    case u =>
      log.debug(s"unknown message $u")
  }

  def forwardHandle(wire: Wiring): Receive = {
    case Forward(v) =>
      val result = gate.module(v)
      wire.next ! Forward(result)
      context become backwardHandle(v, result, wire)
  }

  def backwardHandle(input: Variable, output: Variable, wire: Wiring): Receive = {
    case Backward(g) =>
      log.debug(s"receive backward g shape: ${g.shape}")
      optimizer.zeroGrad()
      output.backward(g)
      optimizer.step()
      wire.prev ! Backward(input.grad)
      context become forwardHandle(wire)
  }
}

object Gate {

  def stage(g: Gate)(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new GateActor(g)))

  case class Forward(v: Variable)
  case class Backward(v: Variable)
}
