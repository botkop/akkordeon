package botkop.akkordeon.flipflop

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import botkop.akkordeon.{DataIterator, Stageable}
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

  var wire: Wire = _
  var epoch = 0
  var trainingLoss = 0.0 // training loss of an epoch
  var epochStartTime = 0L
  var epochDuration = 0L

  override def receive: Receive = {
    case w: Wire =>
      log.debug(s"received wire $w")
      wire = w
      val di = tdl.toIterator
      context become beginPoint(di, di.next())

    case u =>
      log.error(s"unknown message $u")
  }

  def beginPoint(di: DataIterator,
                 batch: (Variable, Variable),
                 cumLoss: Double = 0): Receive = {
    case msg @ (Start | _: Backward) =>
      if (msg == Start) {
        epoch += 1
        epochStartTime = System.currentTimeMillis()
      }
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
        // end of epoch
        // store training loss for this epoch
        trainingLoss = newLoss / tdl.numBatches
        epochDuration = (System.currentTimeMillis() - epochStartTime) / 1000
        evaluate()
      }
  }

  def evaluate(): Unit = {
    val vi = vdl.iterator
    val (x, y) = vi.next()
    wire.next ! Eval(x)
    context become evalHandler(y, vi)
  }

  def evalHandler(y: Variable,
                  vi: DataIterator,
                  cumLoss: Double = 0,
                  cumEval: Double = 0): Receive = {

    case Eval(x) =>
      val l = cumLoss + loss(x, y).data.squeeze()
      val e = cumEval + evaluator(x, y)
      if (vi.hasNext) {
        val (x, y) = vi.next()
        wire.next ! Eval(x)
        context become evalHandler(y, vi, l, e)
      } else {
        val valLoss = l / vdl.numBatches
        val eval = e / vdl.numBatches
        println(
          f"epoch: $epoch%5d trn_loss: $trainingLoss%9.6f val_loss: $valLoss%9.6f eval: $eval%9.6f duration: ${epochDuration}s")

        val di = tdl.toIterator
        self ! Start
        context become beginPoint(di, di.next())
      }

    case _: Backward => // ignore

    case u =>
      log.error(s"evalHandler: unknown message $u")
  }

}
