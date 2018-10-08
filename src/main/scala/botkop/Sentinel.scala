package botkop

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import botkop.Gate.{Backward, Forward}
import botkop.Sentinel.Start
import scorch.autograd.Variable
import scorch.data.loader.DataLoader

case class Sentinel(dl: DataLoader,
                    loss: (Variable, Variable) => Variable,
                    evaluator: (Variable, Variable, Double) => Unit)

class SentinelActor(sentinel: Sentinel) extends Actor with ActorLogging {

  import sentinel._

  log.debug("instantiating sentinel actor")

  override def receive: Receive = {
    case wire: Wiring =>
      log.debug(s"received wire $wire")
      val di = dl.toIterator
      context become beginPoint(wire, di, di.next())

    case u =>
      log.debug(s"unknown message $u")
  }

  def beginPoint(wire: Wiring,
                 di: Iterator[(Variable, Variable)],
                 batch: (Variable, Variable)): Receive = {
    case msg @ (Start | _: Backward) =>
      val (x, y) = batch
      wire.next ! Forward(x)

      if (di.hasNext) {
        context become endPoint(wire, y, di, di.next())
      } else {
        // log.info(s"new epoch")
        val ndi = dl.toIterator
        context become endPoint(wire, y, ndi, ndi.next())
      }
    case u =>
      log.error(s"unknown message $u")
  }

  def endPoint(wire: Wiring,
               y: Variable,
               di: Iterator[(Variable, Variable)],
               batch: (Variable, Variable)): Receive = {
    case Forward(yHat) =>
      log.debug("received forward")
      val l = loss(yHat, y)
      evaluator(yHat, y, l.data.squeeze())
      l.backward()

      log.debug(s"yHat.grad shape = ${yHat.grad.shape}")
      wire.prev ! Backward(yHat.grad)
      context become beginPoint(wire, di, batch)
  }
}

object Sentinel {
  case object Start

  def stage(s: Sentinel)(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new SentinelActor(s)))
}
