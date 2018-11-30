package botkop.akkordeon.mkl

import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import akka.dispatch.MessageDispatcher
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.optim.OptimMethod
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

case class Gate[T](module: TensorModule[T],
                   optimizer: OptimMethod[T],
                   name: String)(implicit ev: TensorNumeric[T])
    extends Stageable {
  def stage(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new GateActor(this)), name)
}

class GateActor[T](gate: Gate[T])(implicit ev: TensorNumeric[T]) extends Actor {

  import gate._

  implicit val dispatcher: MessageDispatcher =
    context.system.dispatchers.lookup("bulk-head-dispatcher")

  module.training()
  var wire: Wire = _

  override def receive: Receive = ???

  def messageHandler(activations: Map[Int, (Tensor[T], Tensor[T])]): Receive = {

    case Forward(sentinel, x: Tensor[T], y: Tensor[T], id) =>
      val result = module.forward(x).toTensor[T]
      wire.next.getOrElse(sentinel) ! Forward(sentinel, result, y, id)
      context become messageHandler(activations + (id -> (x, result)))

  }
}
