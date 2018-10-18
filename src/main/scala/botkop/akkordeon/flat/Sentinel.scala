package botkop.akkordeon.flat

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import botkop.akkordeon.{DataIterator, Stageable}
import scorch.autograd.Variable

case class Sentinel(trainingDataLoader: ActorRef,
                    validationDataLoader: ActorRef,
                    loss: (Variable, Variable) => Variable,
                    evaluator: (Variable, Variable) => Double,
                    name: String)
    extends Stageable {
  def stage(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new SentinelActor(this)), name)
}

class SentinelActor(sentinel: Sentinel) extends Actor with ActorLogging {

  import sentinel._

  var wire: Wire = _

  var trainingLoss = 0.0
  var numTrainingBatches = 0
  var validationLoss = 0.0
  var validationScore = 0.0

  override def receive: Receive = {
    case w: Wire =>
      log.debug(s"received wire $w")
      wire = w
      context become trainingHandler

    case u =>
      log.error(s"receive: unknown message ${u.getClass.getName}")
  }

  def trainingHandler: Receive = {
    case Start =>
      (1 to 4).foreach { _ =>
        trainingDataLoader ! NextBatch
      }

    case Batch(x, y) =>
      wire.next ! Forward(x, y)

    case Forward(yHat, y) =>
      val l = loss(yHat, y)
      l.backward()
      wire.prev ! Backward(yHat.grad)
      trainingLoss += l.data.squeeze()
      numTrainingBatches += 1

    case Backward(_) =>
      trainingDataLoader ! NextBatch

    case Epoch(epoch, duration) =>
      trainingLoss /= numTrainingBatches
      log.info(
        f"epoch: $epoch%5d " +
          f"trn_loss: $trainingLoss%9.6f " +
          f"duration: ${duration}s")
      numTrainingBatches = 0
      trainingLoss = 0

    case u =>
      log.error(s"trainingHandler: unknown message ${u.getClass.getName}")

  }

}
