package botkop.akkordeon.multi

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import botkop.akkordeon.{DataIterator, Stageable}
import botkop.multi._
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

  var wire: MultiWire = _
  var supervisor: ActorRef = _

  override def receive: Receive = {
    case w: MultiWire =>
      log.debug(s"received wire $w")
      wire = w
      supervisor = sender()
    case Start =>
      val vi = vdl.toIterator
      val (x, y) = vi.next()
      for (w <- wire.next) { w ! Eval(x) }
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
        for (w <- wire.next) { w ! Eval(x) }
        context become endState(y: Variable, vi, vi.next(), l, e)
      } else {
        supervisor ! EvalResult(l / vdl.numBatches, e / vdl.numBatches)
        context become receive
      }
  }
}
