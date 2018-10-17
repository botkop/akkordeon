package botkop.akkordeon.flat

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

  // var testDataIterator: DataIterator = testDataLoader.iterator
  var testDataIterator: DataIterator = _
  var validationDataIterator: DataIterator = _

  override def receive: Receive = {
    case w: Wire =>
      log.debug(s"received wire $w")
      wire = w
      // context become startPoint(testDataIterator.next())
      context become handler

    case u =>
      log.error(s"unknown message $u")
  }


  def handler(): Receive = {

    case Start =>
      testDataIterator = testDataLoader.iterator
      epoch += 1
      epochStartTime = System.currentTimeMillis()
      trainingLoss = 0

      val (x, y) = testDataIterator.next()
      wire.next ! Forward(x, y)

    case Forward(yHat, y) =>
      val l = loss(yHat, y)
      l.backward()
      wire.prev ! Backward(yHat.grad)
      trainingLoss += l.data.squeeze()

    case _: Backward =>
      if (testDataIterator.hasNext) {
        val (x, y) = testDataIterator.next()
        wire.next ! Forward(x, y)
      } else {

        epochDuration = (System.currentTimeMillis() - epochStartTime) / 1000
        trainingLoss /= testDataLoader.numBatches

        println(s"$epochDuration     $trainingLoss")

        self ! Start
      }

    case Validate(x, y) =>
      validationLoss += loss(x, y).data.squeeze()
      validationScore += evaluator(x, y)

      if (validationDataIterator.hasNext) {
        val (x, y) = validationDataIterator.next()
        wire.next ! Validate(x, y)
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
      }

    case u =>
      log.error(s"beginPoint: unknown message $u")
  }

//  def startPoint(batch: (Variable, Variable)): Receive = {
//    case msg @ (Start | _: Backward) =>
//      if (msg == Start) { // new epoch
//        epoch += 1
//        epochStartTime = System.currentTimeMillis()
//        trainingLoss = 0
//      }
//      val (x, y) = batch
//      wire.next ! Forward(x, y)
//      context become endPoint
//
//    case u =>
//      log.error(s"beginPoint: unknown message $u")
//  }
//
//
//  def endPoint: Receive = {
//    case Forward(yHat, y) =>
//      val l = loss(yHat, y)
//      l.backward()
//      wire.prev ! Backward(yHat.grad)
//      trainingLoss += l.data.squeeze()
//
//      if (testDataIterator.hasNext) {
//        context become startPoint
//      } else {
//        endOfEpoch()
//      }
//  }

//  def endOfEpoch(): Unit = {
//    epochDuration = (System.currentTimeMillis() - epochStartTime) / 1000
//    trainingLoss /= testDataLoader.numBatches
//
//    validationLoss = 0
//    validationScore = 0
//
//    validationDataIterator = validationDataLoader.iterator
//    val (x, y) = validationDataIterator.next()
//    wire.next ! Validate(x, y)
//    context become validationHandler
//
//  }
//
//  def validationHandler: Receive = {
//
//    case Validate(x, y) =>
//      validationLoss += loss(x, y).data.squeeze()
//      validationScore += evaluator(x, y)
//
//      if (validationDataIterator.hasNext) {
//        val (x, y) = validationDataIterator.next()
//        wire.next ! Validate(x, y)
//        context become validationHandler
//      } else {
//        validationLoss /= validationDataLoader.numBatches
//        validationScore /= validationDataLoader.numBatches
//
//        println(
//          f"epoch: $epoch%5d " +
//            f"trn_loss: $trainingLoss%9.6f " +
//            f"val_loss: $validationLoss%9.6f " +
//            f"score: $validationScore%9.6f " +
//            f"duration: ${epochDuration}s")
//
//        testDataIterator = testDataLoader.iterator
//        self ! Start
//         context become startPoint(testDataIterator.next())
//        context become startPoint
//      }
//
//    case _: Backward => // ignore, this is the backprop of the last forward message
//
//    case u =>
//      log.error(s"evalHandler: unknown message $u")
//  }

}
