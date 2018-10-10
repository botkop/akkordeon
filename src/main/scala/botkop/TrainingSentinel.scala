package botkop

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import botkop.Gate.{Backward, Forward}
import botkop.Sentinel.Start
import scorch.autograd.Variable
import scorch.data.loader.DataLoader

case class TrainingSentinel(tdl: DataLoader,
                            loss: (Variable, Variable) => Variable)
    extends Stageable {
  def stage(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new TrainingSentinelActor(this)), "trainingSentinel")
}

class TrainingSentinelActor(sentinel: TrainingSentinel)
    extends Actor
    with ActorLogging {
  import sentinel._

  var wire: Wiring = _
  var supervisor: ActorRef = _

  override def receive: Receive = {
    case initWire: Wiring =>
      log.debug(s"received wire $initWire")
      wire = initWire
      supervisor = sender()

    case Start =>
      val di = tdl.toIterator
      val (x, y) = di.next()
      wire.next ! Forward(x)
      context become endPoint(y, di, 0)

    case u =>
      log.error(s"unknown message $u")
  }

  def beginPoint(di: DataIterator,
                 batch: (Variable, Variable),
                 cumLoss: Double): Receive = {
    case _: Backward =>
      val (x, y) = batch
      wire.next ! Forward(x)
      context become endPoint(y, di, cumLoss)

    case u =>
      log.error(s"beginPoint: unknown message $u")
  }

  def endPoint(y: Variable, di: DataIterator, cumLoss: Double): Receive = {
    case Forward(yHat) =>
      val l = loss(yHat, y)
      l.backward()
      wire.prev ! Backward(yHat.grad)
      val newLoss = cumLoss + l.data.squeeze()

      if (di.hasNext) {
        context become beginPoint(di, di.next(), newLoss)
      } else {
        context become receive
      }

    case u =>
      log.error(s"endPoint: unknown message $u")
  }
}
