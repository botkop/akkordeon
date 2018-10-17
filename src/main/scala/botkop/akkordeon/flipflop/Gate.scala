package botkop.akkordeon.flipflop

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import botkop.akkordeon.Stageable
import scorch.autograd.Variable
import scorch.nn.Module
import scorch.optim.Optimizer

case class Gate(module: Module, optimizer: Optimizer, name: String)
    extends Stageable {
  def stage(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new GateActor(this)), name)
}

class GateActor(gate: Gate) extends Actor with ActorLogging {

  import gate._

  var wire: Wire = _

  override def receive: Receive = {
    case w: Wire =>
      log.debug(s"received wire $w")
      this.wire = w
      context become forwardHandle
    case u =>
      log.error(s"unknown message $u")
  }

  def forwardHandle: Receive = {
    case Forward(x) =>
      val result = module(x)
      wire.next ! Forward(result)
      context become backwardHandle(x, result)
    case Validate(x) =>
      wire.next ! Validate(module(x))
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
    case Validate(x) =>
      wire.next ! Validate(module(x))
    case u =>
      log.error(s"unknown message $u")
  }
}

