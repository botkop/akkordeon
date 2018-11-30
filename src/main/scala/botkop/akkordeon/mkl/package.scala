package botkop.akkordeon

import akka.actor.ActorRef
import com.intel.analytics.bigdl.tensor.Tensor

import scala.util.Random

package object mkl {

  case object Start

  trait Message
  case class Forward[T](sentinel: ActorRef,
                        x: Tensor[T],
                        y: Tensor[T],
                        id: Int = Random.nextInt)
      extends Message

  case class Backward[T](sentinel: ActorRef, g: Tensor[T], id: String)
      extends Message

  case class Validate[T](sentinel: ActorRef, x: Tensor[T], y: Tensor[T])
      extends Message

  case class Batch[T](x: Tensor[T], y: Tensor[T])

  case class NextBatch(recipient: ActorRef)

  case class Epoch(provider: String, n: Int, duration: Long)

}
