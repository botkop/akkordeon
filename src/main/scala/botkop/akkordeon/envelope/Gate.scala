package botkop.akkordeon.envelope

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import botkop.akkordeon.Stageable
import scorch.autograd.Variable
import scorch.nn.Module
import scorch.optim.Optimizer

import scala.concurrent.Future

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
    case Forward(v, sentinel) =>
      val result = module(v)
      wire.next.getOrElse(sentinel) ! Forward(result, sentinel)
      context become backwardHandle(v, result)
    case Eval(x, sentinel) =>
      wire.next.getOrElse(sentinel) ! Eval(module(x), sentinel)
    case u =>
      log.error(s"unknown message $u")
  }

  def backwardHandle(input: Variable, output: Variable): Receive = {
    case Backward(g, sentinel) =>
      optimizer.zeroGrad()
      output.backward(g)
      wire.prev.getOrElse(sentinel) ! Backward(input.grad, sentinel)
      Future(optimizer.step())(context.dispatcher)
      context become forwardHandle
    case Eval(x, sentinel) =>
      wire.next.getOrElse(sentinel) ! Eval(module(x), sentinel)
    case u =>
      log.error(s"unknown message $u")
  }

}
