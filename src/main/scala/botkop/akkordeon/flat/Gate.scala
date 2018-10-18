package botkop.akkordeon.flat

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
      context become messageHandler(List.empty)
    case u =>
      log.error(s"unknown message $u")
  }

  def messageHandler(activations: List[(Variable, Variable)]): Receive = {

    case Validate(x) =>
      wire.next ! Validate(module(x))

    case Forward(x, y) =>
      val result = module(x)
      wire.next ! Forward(result, y)
      context become messageHandler(activations :+ (x, result))

    case Backward(g) =>
      activations match {
        case (input, output) :: tail =>
          // todo average grads, or divide learning rate by parallelism ?
          optimizer.zeroGrad()
          output.backward(g)
          wire.prev ! Backward(input.grad)
          optimizer.step()
          context become messageHandler(tail)
        case _ =>
          log.error("backward message received but no activations registered")
      }

    case u =>
      log.error(s"unknown message $u")
  }

}
