package botkop.akkordeon.mkl

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.optim.ValidationMethod

import scala.collection.mutable.ArrayBuffer

case class Sentinel[@specialized(Float, Double) T](dp: DataProvider[T],
                       concurrency: Int,
                       criterion: TensorCriterion[T],
                       scoreFunctions: List[ValidationMethod[T]],
                       name: String)
    extends Stageable {
  def stage(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new SentinelActor(this)), name)
}

class SentinelActor[@specialized(Float, Double) T](sentinel: Sentinel[T]) extends Actor with ActorLogging {

  import sentinel._

  var loss = 0.0
  val scores: ArrayBuffer[Double] = ArrayBuffer.fill(scoreFunctions.length)(0.0)
  var numBatches = 0

  val dl: ActorRef = dp.stage(context.system)

  override def receive: Receive = {
    case w: Wire =>
      log.debug(s"received wire $w")
      context become messageHandler(w.prev.get, w.next.get)

    case u =>
      log.error(s"receive: unknown message ${u.getClass.getName}")
  }


  def messageHandler(prev: ActorRef, next: ActorRef): Receive = {

    case Start =>
      1 to concurrency foreach (_ => dl ! NextBatch(next))

    case Forward(ar, yHat, y, id) =>
      val l = criterion.forward(yHat, y)
      val g = criterion.backward(yHat, y)
      prev ! Backward(ar, g, id)
      loss += l.asInstanceOf[Float]
      scoreFunctions.zipWithIndex.foreach {
        case (f, i) =>
          // todo make better
          scores(i) += f(yHat, y).result()._1
      }
      numBatches += 1

    case _: Backward =>
      dl ! NextBatch(next)

    case Validate(_, yHat, y) =>
      numBatches += 1
      loss += criterion.forward(yHat, y).asInstanceOf[Float]
      scoreFunctions.zipWithIndex.foreach {
        case (f, i) =>
          scores(i) += f(yHat, y).result()._1
      }
      if (numBatches < dp.numBatches)
        dl ! NextBatch(next)

    case Epoch(epochName, epoch, duration) =>
      loss /= numBatches
      scores.indices.foreach(i => scores(i) /= numBatches)
      println(
        f"$epochName%-10s epoch: $epoch%5d " +
          f"loss: $loss%9.6f " +
          f"duration: ${duration}ms " + scores)
      loss = 0
      numBatches = 0
      scores.indices.foreach(i => scores(i) = 0)

    case u =>
      log.error(s"messageHandler: unknown message ${u.getClass.getName}")
  }
}
