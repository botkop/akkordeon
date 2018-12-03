package botkop.akkordeon.mkl

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import akka.dispatch.MessageDispatcher
import com.intel.analytics.bigdl.nn.Sequential
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.concurrent.{ExecutionContextExecutor, Future}

case class Gate[T](module: Sequential[T],
                   optimizer: Optimizer[T],
                   name: String)(implicit ev: TensorNumeric[T])
    extends Stageable {
  def stage(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new GateActor(this)), name)
}

class GateActor[T](gate: Gate[T])(implicit ev: TensorNumeric[T])
    extends Actor
    with ActorLogging {

  import gate._

//  implicit val dispatcher: MessageDispatcher =
//    context.system.dispatchers.lookup("bulk-head-dispatcher")

  implicit val dispatcher: ExecutionContextExecutor =
    context.dispatcher

  module.training()
  var wire: Wire = _

  override def receive: Receive = {
    case w: Wire =>
      log.debug(s"received wire $w")
      this.wire = w
      context become messageHandler(Map.empty)
    case u =>
      log.error(s"$name: receive: unknown message ${u.getClass.getName}")
  }

  def messageHandler(activations: Map[String, Tensor[T]]): Receive = {

    case Forward(sentinel, x: Tensor[T], y: Tensor[T], id) =>
      val result = module.forward(x)
      wire.next.getOrElse(sentinel) ! Forward(sentinel, result, y, id)
      // fucking keeps state
      context become messageHandler(activations + (id -> x))

    case Backward(sentinel, g, id) =>
      val input = activations(id)
//      Future {
        module.zeroGradParameters()
        val grad = module.backward(input, g)
        wire.prev.getOrElse(sentinel) ! Backward(sentinel, grad, id)
        optimizer.step()
//      }
      context become messageHandler(activations - id)

    case Validate(sentinel, x, y) =>
      Future {
//        module.evaluate()
        wire.next.getOrElse(sentinel) ! Validate(sentinel, module.forward(x), y)
//        module.training()
      }

  }

//  def optimize(): Unit = {
//    val (ps, dps) = module.parameters()
//    ps.zip(dps).foreach {
//      case (p, dp) =>
//        optimizer.optimize(_ => (ev.zero, dp), p)
//    }
//  }
}
