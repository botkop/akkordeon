package botkop.akkordeon

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
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

  import Gate._
  import gate._

  var wire: Wiring = _

  override def receive: Receive = {
    case initWire: Wiring =>
      log.debug(s"received wire $initWire")
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
      Future(optimizer.step())(context.dispatcher)
      context become forwardHandle
    case Eval(x, y) =>
      wire.next ! Eval(module(x), y)
    case u =>
      log.error(s"unknown message $u")
  }
}

object Gate {
  case class Forward(v: Variable)
  case class Backward(v: Variable)
  case class Eval(x: Variable, y: Variable)
  object Eval {
    def apply(xy: (Variable, Variable)): Eval = Eval(xy._1, xy._2)
  }
}
