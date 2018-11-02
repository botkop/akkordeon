package botkop.akkordeon.single

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

  var wire: Wire = _

  var loss = 0.0
  val scores: ArrayBuffer[Double] = ArrayBuffer.fill(scoreFunctions.length)(0.0)
  var numBatches = 0

  val dl: ActorRef = dataProvider.stage(context.system)

  lazy val nextBatch = NextBatch(wire.next.get)

  override def receive: Receive = {
    case w: Wire =>
      log.debug(s"received wire $w")
      wire = w
      context become messageHandler

    case u =>
      log.error(s"receive: unknown message ${u.getClass.getName}")
  }

  def messageHandler: Receive = {
    case Start =>
      1 to concurrency foreach (_ => dl ! nextBatch)

    case Forward(ar, yHat, y, id) =>
      val l = lossFunction(yHat, y)
      l.backward()
      wire.prev.get ! Backward(ar, yHat.grad, id)
      loss += l.data.squeeze()
      numBatches += 1

    case _: Backward =>
      dl ! nextBatch

    case Validate(_, x, y) =>
      numBatches += 1
      loss += lossFunction(x, y).data.squeeze()
      if (numBatches < dataProvider.dl.numBatches)
        dl ! nextBatch

    case Epoch(epochName, epoch, duration) =>
      loss /= numBatches
      println(
        f"$epochName%-10s epoch: $epoch%5d " +
          f"loss: $loss%9.6f " +
          f"duration: ${duration}ms")
      loss = 0
      numBatches = 0

    case u =>
      log.error(s"messageHandler: unknown message ${u.getClass.getName}")
  }

}
