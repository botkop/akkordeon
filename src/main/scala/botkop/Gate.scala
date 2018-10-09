package botkop

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import scorch.autograd.Variable
import scorch.nn.Module
import scorch.optim.Optimizer

case class Gate(module: Module, optimizer: Optimizer)

class GateActor(gate: Gate) extends Actor with ActorLogging {

  import gate._
  import botkop.Gate._

  var wire: Wiring = _

  override def receive: Receive = {
    case initWire: Wiring =>
      log.debug(s"received wire $wire")
      this.wire = initWire
      context become forwardHandle
    case u =>
      log.error(s"unknown message $u")
  }

  def forwardHandle: Receive = {
    case Forward(v) =>
      val result = module(v)
      wire.next ! Forward(result)
      context become backwardHandle(v, result)
    case Eval(x, y) =>
      wire.next ! Eval(module(x), y)
    case u =>
      log.error(s"unknown message $u")
  }

  def backwardHandle(input: Variable, output: Variable): Receive = {
    case Backward(g) =>
      optimizer.zeroGrad()
      output.backward(g)
      wire.prev ! Backward(input.grad)
      optimizer.step()
      context become forwardHandle
    case Eval(x, y) =>
      wire.next ! Eval(module(x), y)
    case u =>
      log.error(s"unknown message $u")
  }
}

object Gate {

  def stage(g: Gate)(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new GateActor(g)))

  case class Forward(v: Variable)
  case class Backward(v: Variable)
  case class Eval(x: Variable, y: Variable)
  object Eval {
    def apply(xy: (Variable, Variable)): Eval = Eval(xy._1, xy._2)
  }
}
