package botkop.akkordeon.hash

import java.util.UUID

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import scorch.autograd.Variable
import scorch.data.loader.DataLoader

import scala.util.Random

case class TrainingComponents(trainingDataLoader: DataLoader,
                              trainingConcurrency: Int,
                              loss: (Variable, Variable) => Variable)

case class ValidationComponents(validationDataLoader: DataLoader,
                                validationConcurrency: Int,
                                evaluator: (Variable, Variable) => Double)

case class Sentinel(trainer: TrainingComponents,
                    validator: Option[ValidationComponents],
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

  val dptn = s"$name-training"
  val dpvn = s"$name-validation"

  val tdl: ActorRef =
    DataProvider(trainer.trainingDataLoader, dptn).stage(context.system)
  val vdl: Option[ActorRef] = validator.map { v =>
    DataProvider(v.validationDataLoader, dpvn).stage(context.system)
  }

  private val b2t = (b: Batch) => {
    val id = Random.nextString(4)
    Forward(id, self, b.x, b.y)
  }
  lazy val nextTrainingBatch = NextBatch(wire.next.get, b2t)

  private val b2v = (b: Batch) => Validate(self, b.x, b.y)
  lazy val nextValidationBatch = NextBatch(wire.next.get, b2v)

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
      1 to trainer.trainingConcurrency foreach (_ => tdl ! nextTrainingBatch)
      if (validator.isDefined) {
        1 to validator.get.validationConcurrency foreach (_ =>
          vdl.get ! nextValidationBatch)
      }

    case Forward(id, ar, yHat, y) =>
      val l = trainer.loss(yHat, y)
      l.backward()
      wire.prev.get ! Backward(id, ar, yHat.grad)
      trainingLoss += l.data.squeeze()
      numTrainingBatches += 1

    case _: Backward =>
      tdl ! nextTrainingBatch

    case Validate(_, x, y) if validator.isDefined =>
      numValidationBatches += 1
      validationLoss += trainer.loss(x, y).data.squeeze()
      validationScore += validator.get.evaluator(x, y)
      if (numValidationBatches < validator.get.validationDataLoader.numBatches)
        vdl.get ! nextValidationBatch

    case Epoch(epochName, epoch, duration) if validator.isDefined =>
      if (epochName == dptn) {
        trainingLoss /= numTrainingBatches
        validationLoss /= numValidationBatches
        validationScore /= numValidationBatches
        println(
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
        1 to validator.get.validationConcurrency foreach (_ =>
          vdl.get ! nextValidationBatch)
      }

    case u =>
      // log.error(s"messageHandler: unknown message ${u.getClass.getName}")
  }

}
