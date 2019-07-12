package botkop.akkordeon

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import scorch.autograd.Variable
import scorch.data.loader.DataLoader

case class DataProvider(dl: DataLoader,
                        f: (Batch, ActorRef) => Message,
                        name: String)
  extends Stageable {

  override def stage(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new DataProviderActor(this)), name)
}

object DataProvider {
  val b2f: (Batch, ActorRef) => Forward = (b, sentinel) =>
    Forward(sentinel, b.x, b.y)

  val b2v: (Batch, ActorRef) => Validate = (b, sentinel) =>
    Validate(sentinel, b.x, b.y)

  def apply(dataSet: String,
            mode: String,
            miniBatchSize: Int,
            take: Option[Int] = None,
            name: String): DataProvider = {
    val dl = DataLoader.instance(dataSet, mode, miniBatchSize, take)
    DataProvider(dl, name)
  }

  def apply(dl: DataLoader, name: String): DataProvider = {
    val f = dl.mode match {
      case "train" => b2f
      case "validate" => b2v
    }
    DataProvider(dl, f, name)
  }

}

class DataProviderActor(dp: DataProvider) extends Actor with ActorLogging {

  import dp._

  override def receive: Receive = {
    val ndi = dl.iterator
    provide(ndi, ndi.next(), epoch = 0, startTime = 0)
  }

  def nextMessage(di: DataIterator,
                  epoch: Int,
                  startTime: Long): Unit = {
    if (di.hasNext) {
      context become provide(di, di.next(), epoch, startTime)
    } else {
      val duration = System.nanoTime() - startTime
      sender() ! Epoch(name, epoch, duration)
      val ndi = dl.iterator
      context become provide(ndi, ndi.next(), epoch + 1, System.nanoTime())
    }
  }

  def provide(di: DataIterator,
              nextBatch: (Variable, Variable),
              epoch: Int,
              startTime: Long): Receive = {

    case FirstBatch(r) =>
      r forward f(Batch(nextBatch), sender())
      nextMessage(di, epoch, startTime)
      // context become provide(di, di.next(), epoch + 1, System.nanoTime())

    case NextBatch(r) =>
      r forward f(Batch(nextBatch), sender())
      nextMessage(di, epoch, startTime)

//      if (di.hasNext) {
//        context become provide(di, di.next(), epoch, startTime)
//      } else {
//        val duration = System.nanoTime() - startTime
//        sender() ! Epoch(name, epoch, duration)
//        val ndi = dl.iterator
//        context become provide(ndi, ndi.next(), epoch + 1, System.nanoTime())
//      }
  }
}
