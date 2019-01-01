package botkop.akkordeon

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import scorch.autograd.Variable

import scala.collection.mutable.ArrayBuffer

case class Sentinel(dataProvider: DataProvider,
                    concurrency: Int,
                    lossFunction: (Variable, Variable) => Variable,
                    scoreFunctions: List[(Variable, Variable) => Double],
                    name: String)
    extends Stageable {
  def stage(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new SentinelActor(this)), name)
}

class SentinelActor(sentinel: Sentinel) extends Actor with ActorLogging {

  import sentinel._

  var loss = 0.0
  val scores: ArrayBuffer[Double] = ArrayBuffer.fill(scoreFunctions.length)(0.0)
  var numBatches = 0

  val dl: ActorRef = dataProvider.stage(context.system)

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
      val l = lossFunction(yHat, y)
      l.backward()
      prev ! Backward(ar, yHat.grad, id)
      loss += l.data.squeeze()
      scoreFunctions.zipWithIndex.foreach {
        case (f, i) =>
          scores(i) += f(yHat, y)
      }
      numBatches += 1

    case _: Backward =>
      dl ! NextBatch(next)

    case Validate(_, yHat, y) =>
      numBatches += 1
      loss += lossFunction(yHat, y).data.squeeze()
      scoreFunctions.zipWithIndex.foreach {
        case (f, i) =>
          scores(i) += f(yHat, y)
      }
      if (numBatches < dataProvider.dl.numBatches)
        dl ! NextBatch(next)

    case Epoch(epochName, epoch, duration) =>
      loss /= numBatches
      scores.indices.foreach(i => scores(i) /= numBatches)
      log.info(
        f"$epochName%-10s epoch: $epoch%5d " +
          f"loss: $loss%9.6f " +
          s"""scores: (${scores.mkString(", ")}) """ +
          f"duration: ${duration / 1e6.toLong}ms")

      loss = 0
      numBatches = 0
      scores.indices.foreach(i => scores(i) = 0)

    case u =>
      log.error(s"messageHandler: unknown message ${u.getClass.getName}")
  }

}
