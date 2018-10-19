package botkop.akkordeon.wheels

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import botkop.akkordeon.Stageable
import scorch.autograd.Variable
import scorch.data.loader.DataLoader

case class Sentinel(trainingDataLoader: DataLoader,
                    validationDataLoader: DataLoader,
                    trainingConcurrency: Int,
                    validationConcurrency: Int,
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
  var numValidationBatches = 0

  val tdl: ActorRef =
    DataProvider(trainingDataLoader, "training").stage(context.system)
  val vdl: ActorRef =
    DataProvider(validationDataLoader, "validation").stage(context.system)

  private val b2t = (b: Batch) => Forward(b.x, b.y)
  lazy val nextTrainingBatch = NextBatch(wire.next, b2t)

  private val b2v = (b: Batch) => Validate(b.x, b.y)
  lazy val nextValidationBatch = NextBatch(wire.next, b2v)

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
      1 to trainingConcurrency foreach (_ => tdl ! nextTrainingBatch)
      1 to validationConcurrency foreach (_ => vdl ! nextValidationBatch)

    case Forward(yHat, y) =>
      val l = loss(yHat, y)
      l.backward()
      wire.prev ! Backward(yHat.grad)
      trainingLoss += l.data.squeeze()
      numTrainingBatches += 1

    case Backward(_) =>
      tdl ! nextTrainingBatch

    case Validate(x, y) =>
      numValidationBatches += 1
      validationLoss += loss(x, y).data.squeeze()
      validationScore += evaluator(x, y)
      if (numValidationBatches < validationDataLoader.numBatches)
        vdl ! nextValidationBatch

    case Epoch("training", epoch, duration) =>
      trainingLoss /= numTrainingBatches
      validationLoss /= numValidationBatches
      validationScore /= numValidationBatches
      log.info(
        f"epoch: $epoch%5d " +
          f"trn_loss: $trainingLoss%9.6f " +
          f"val_loss: $validationLoss%9.6f " +
          f"val_score: $validationScore%9.6f " +
          f"duration: ${duration}ms")
      numTrainingBatches = 0
      trainingLoss = 0
      numValidationBatches = 0
      validationLoss = 0
      validationScore = 0
      1 to validationConcurrency foreach (_ => vdl ! nextValidationBatch)

    case Epoch("validation", _, _) => // ignore

    case u =>
      log.error(s"trainingHandler: unknown message ${u.getClass.getName}")
  }

}
