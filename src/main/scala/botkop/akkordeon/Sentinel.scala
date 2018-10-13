package botkop.akkordeon

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import scorch.autograd.Variable
import scorch.data.loader.DataLoader

case class Sentinel(tdl: DataLoader,
                    vdl: DataLoader,
                    loss: (Variable, Variable) => Variable,
                    evaluator: (Variable, Variable) => Double,
                    name: String)
    extends Stageable {
  def stage(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new SentinelActor(this)), name)
}

class SentinelActor(sentinel: Sentinel) extends Actor with ActorLogging {

  import sentinel._

  var epoch = 0
  var wire: Wiring = _

  override def receive: Receive = {
    case initWire: Wiring =>
      log.debug(s"received wire $initWire")
      wire = initWire
      val di = tdl.toIterator
      context become beginPoint(di, di.next())

    case u =>
      log.error(s"unknown message $u")
  }

  def beginPoint(di: DataIterator,
                 batch: (Variable, Variable),
                 cumLoss: Double = 0): Receive = {
    case msg @ (Start | _: Backward) =>
      if (msg == Start) epoch = epoch + 1
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
        val vi = vdl.iterator
        wire.next ! Eval(vi.next())
        val avgLoss = newLoss / tdl.numBatches
        context become evalHandler(vi, avgLoss)
      }
  }

  def evalHandler(vi: DataIterator,
                  trnLoss: Double,
                  cumLoss: Double = 0,
                  cumEval: Double = 0): Receive = {

    case Eval(x, y) =>
      val l = cumLoss + loss(x, y).data.squeeze()
      val e = cumEval + evaluator(x, y)
      if (vi.hasNext) {
        wire.next ! Eval(vi.next())
        context become evalHandler(vi, trnLoss, l, e)
      } else {
        val valLoss = l / vdl.numBatches
        val eval = e / vdl.numBatches
        println(
          f"epoch: $epoch%5d trn_loss: $trnLoss%9.6f val_loss: $valLoss%9.6f eval: $eval%9.6f")

        val di = tdl.toIterator
        self ! Start
        context become beginPoint(di, di.next())
      }

    case _: Backward =>
    // ignore

    case u =>
      log.error(s"evalHandler: unknown message $u")
  }
}

