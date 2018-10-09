package botkop

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import botkop.Gate.{Backward, Eval, Forward}
import botkop.Sentinel._
import scorch.autograd.Variable
import scorch.data.loader.DataLoader

case class Sentinel(tdl: DataLoader,
                    vdl: DataLoader,
                    loss: (Variable, Variable) => Variable,
                    evaluator: (Variable, Variable) => Double)

class SentinelActor(sentinel: Sentinel) extends Actor with ActorLogging {

  import sentinel._

  log.debug("instantiating sentinel actor")

  override def receive: Receive = {
    case wire: Wiring =>
      log.debug(s"received wire $wire")
      val di = tdl.toIterator
      context become beginPoint(wire, di, di.next())

    case u =>
      log.error(s"unknown message $u")
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
        val vi = vdl.iterator
        wire.next ! Eval(vi.next())
        context become evalHandler(wire, vi)
      }

    case u =>
      log.error(s"beginPoint: unknown message $u")
  }

  def endPoint(wire: Wiring,
               y: Variable,
               di: Iterator[(Variable, Variable)],
               batch: (Variable, Variable)): Receive = {
    case Forward(yHat) =>
      log.debug("received forward")
      val l = loss(yHat, y)
      // println(evaluator(yHat, y))
      l.backward()
      wire.prev ! Backward(yHat.grad)
      context become beginPoint(wire, di, batch)
    case u =>
      log.error(s"endPoint: unknown message $u")
  }

  def evalHandler(wire: Wiring,
                  vi: Iterator[(Variable, Variable)],
                  n: Int = 0,
                  cumLoss: Double = 0,
                  cumEval: Double = 0): Receive = {

    case Eval(x, y) =>
      val l = loss(x, y).data.squeeze()
      val e = evaluator(x, y)
      if (vi.hasNext) {
        wire.next ! Eval(vi.next())
        context become evalHandler(wire, vi, n + 1, cumLoss + l, cumEval + e)
      } else {
        val al = (cumLoss + l) / n
        val ae = (cumEval + e) / n
        println(s"loss: $al, eval: $ae")

//        val ndi = tdl.toIterator
//        val (x, y) = ndi.next()
//        val nn = ndi.next()
//        wire.next ! Forward(x)
//        context become endPoint(wire, y, ndi, nn)

//        val di = tdl.toIterator
//        self ! Start
//        context become beginPoint(wire, di, di.next())

      }

    case u =>
      log.error(s"evalHandler: unknown message $u")
  }
}

object Sentinel {
  case object Start

  def stage(s: Sentinel)(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new SentinelActor(s)))
}
