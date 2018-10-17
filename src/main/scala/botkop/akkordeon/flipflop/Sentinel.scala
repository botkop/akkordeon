package botkop.akkordeon.flipflop

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import botkop.akkordeon.{DataIterator, Stageable}
import scorch.autograd.Variable
import scorch.data.loader.DataLoader

case class Sentinel(testDataLoader: DataLoader,
                    validationDataLoader: DataLoader,
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
  var epoch = 0
  var epochStartTime = 0L
  var epochDuration = 0L

  var trainingLoss = 0.0
  var validationLoss = 0.0
  var validationScore = 0.0

  var testDataIterator: DataIterator = testDataLoader.iterator
  var validationDataIterator: DataIterator = _

  override def receive: Receive = {
    case w: Wire =>
      log.debug(s"received wire $w")
      wire = w
      context become startPoint(testDataIterator.next())

    case u =>
      log.error(s"unknown message $u")
  }

def startPoint(batch: (Variable, Variable)): Receive = {
  case msg @ (Start | _: Backward) =>
    if (msg == Start) { // new epoch
      epoch += 1
      epochStartTime = System.currentTimeMillis()
      trainingLoss = 0
    }
    val (x, y) = batch
    wire.next ! Forward(x)
    context become endPoint(y)

  case u =>
    log.error(s"beginPoint: unknown message $u")
}

  def endPoint(y: Variable): Receive = {
    case Forward(yHat) =>
      val l = loss(yHat, y)
      l.backward()
      wire.prev ! Backward(yHat.grad)
      trainingLoss += l.data.squeeze()

      if (testDataIterator.hasNext) {
        context become startPoint(testDataIterator.next())
      } else {
        endOfEpoch()
      }
  }

  def endOfEpoch(): Unit = {
    epochDuration = (System.currentTimeMillis() - epochStartTime) / 1000
    trainingLoss /= testDataLoader.numBatches

    validationLoss = 0
    validationScore = 0

    validationDataIterator = validationDataLoader.iterator
    val (x, y) = validationDataIterator.next()
    wire.next ! Validate(x)
    context become validationHandler(y)
  }

def validationHandler(y: Variable): Receive = {

  case Validate(x) =>
    validationLoss += loss(x, y).data.squeeze()
    validationScore += evaluator(x, y)

    if (validationDataIterator.hasNext) {
      val (x, y) = validationDataIterator.next()
      wire.next ! Validate(x)
      context become validationHandler(y)
    } else {
      validationLoss /= validationDataLoader.numBatches
      validationScore /= validationDataLoader.numBatches

      println(
        f"epoch: $epoch%5d " +
          f"trn_loss: $trainingLoss%9.6f " +
          f"val_loss: $validationLoss%9.6f " +
          f"score: $validationScore%9.6f " +
          f"duration: ${epochDuration}s")

      testDataIterator = testDataLoader.iterator
      self ! Start
      context become startPoint(testDataIterator.next())
    }

  case _: Backward => // ignore, this is the backprop of the last forward message

  case u =>
    log.error(s"evalHandler: unknown message $u")
}

}
