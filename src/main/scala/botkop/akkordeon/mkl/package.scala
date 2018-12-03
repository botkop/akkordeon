package botkop.akkordeon

import akka.actor.ActorRef
import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.nn.abstractnn.Activity

import scala.util.Random

package object mkl {

  case object Start

  trait Message
  case class Forward(sentinel: ActorRef,
                     x: Activity,
                     y: Activity,
                     id: String = Random.nextString(4))
      extends Message

  case class Backward(sentinel: ActorRef, g: Activity, id: String) extends Message

  case class Validate(sentinel: ActorRef, x: Activity, y: Activity)
      extends Message

  case class Batch(x: Activity, y: Activity)
  object Batch {
    def apply[T](b: MiniBatch[T]): Batch = Batch(b.getInput(), b.getTarget())
  }

  case class NextBatch(recipient: ActorRef)

  case class Epoch(provider: String, n: Int, duration: Long)

}
