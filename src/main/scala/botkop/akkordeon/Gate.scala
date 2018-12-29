package botkop.akkordeon

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import akka.dispatch.MessageDispatcher
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

  implicit val dispatcher: MessageDispatcher =
    context.system.dispatchers.lookup("bulk-head-dispatcher")

  module.inTrainingMode = true
  var wire: Wire = _

  override def receive: Receive = {
    case w: Wire =>
      log.debug(s"received wire $w")
      this.wire = w
      context become messageHandler(Map.empty)
    case u =>
      log.error(s"$name: receive: unknown message ${u.getClass.getName}")
  }

  def messageHandler(activations: Map[Int, (Variable, Variable)]): Receive = {

    case Validate(sentinel, x, y) =>
      Future {
        module.inTrainingMode = false
        wire.next.getOrElse(sentinel) ! Validate(sentinel, module(x), y)
        module.inTrainingMode = true
      }

    case Forward(sentinel, x, y, id) =>
      val result = module(x)
      wire.next.getOrElse(sentinel) ! Forward(sentinel, result, y, id)
      context become messageHandler(activations + (id -> (x, result)))

    case Backward(sentinel, g, id) =>
      val (input, output) = activations(id)
      Future {
        optimizer.zeroGrad()
        output.backward(g)
        wire.prev.getOrElse(sentinel) ! Backward(sentinel, input.grad, id)
        optimizer.step()
      }
      context become messageHandler(activations - id)

    case u =>
      log.error(s"$name: messageHandler: unknown message ${u.getClass.getName}")
  }

}
