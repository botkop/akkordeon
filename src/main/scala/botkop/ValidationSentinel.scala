package botkop

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import botkop.Sentinel.Start
import botkop.ValidationSentinel.{Eval, EvalResult}
import scorch.autograd.Variable
import scorch.data.loader.DataLoader

case class ValidationSentinel(vdl: DataLoader,
                              loss: (Variable, Variable) => Variable,
                              evaluator: (Variable, Variable) => Double)
    extends Stageable {
  def stage(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new ValidationSentinelActor(this)),
                   "validationSentinel")
}

class ValidationSentinelActor(sentinel: ValidationSentinel)
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
      val vi = vdl.toIterator
      val (x, y) = vi.next()
      wire.next ! Eval(x)
      context become endState(y: Variable, vi, vi.next(), 0, 0)
    case u =>
      log.error(s"unknown message $u")
  }

  def endState(y: Variable,
               vi: DataIterator,
               nextBatch: (Variable, Variable),
               cumLoss: Double,
               cumEval: Double): Receive = {
    case Eval(yHat) =>
      val l = cumLoss + loss(yHat, y).data.squeeze()
      val e = cumEval + evaluator(yHat, y)
      if (vi.hasNext) {
        val (x, y) = nextBatch
        wire.next ! Eval(x)
        context become endState(y: Variable, vi, vi.next(), l, e)
      } else {
        supervisor ! EvalResult(l / vdl.numBatches, e / vdl.numBatches)
        context become receive
      }
  }
}

object ValidationSentinel {
  case class Eval(v: Variable)
  case class EvalResult(loss: Double, eval: Double)
}
