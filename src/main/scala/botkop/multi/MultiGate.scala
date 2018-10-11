package botkop.multi

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import botkop.Stageable
import scorch.autograd.Variable
import scorch.nn.Module
import scorch.optim.Optimizer

case class MultiGate(module: Module, optimizer: Optimizer, name: String)
    extends Stageable {
  def stage(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new MultiGateActor(this)), name)
}

class MultiGateActor(gate: MultiGate) extends Actor with ActorLogging {

  import gate._

  var wire: MultiWire = _

  override def receive: Receive = {
    case initWire: MultiWire =>
      log.debug(s"received wire $initWire")
      this.wire = initWire
      context become forwardHandle
    case u =>
      log.error(s"unknown message $u")
  }

  def forwardHandle: Receive = {
    case Forward(v) =>
      val result = module(v)
      wire.next.foreach(_ ! Forward(result))
      context become backwardHandle(v, result)
    case Eval(x) =>
      wire.next.foreach(_ ! Eval(module(x)))
    case u =>
      log.error(s"unknown message $u")
  }

  def backwardHandle(input: Variable, output: Variable): Receive = {
    case Backward(g) =>
      optimizer.zeroGrad()
      output.backward(g)
      wire.prev.foreach(_ ! Backward(input.grad))
      optimizer.step()
      context become forwardHandle
    case Eval(x) =>
      wire.next.foreach(_ ! Eval(module(x)))
    case u =>
      log.error(s"unknown message $u")
  }
}
