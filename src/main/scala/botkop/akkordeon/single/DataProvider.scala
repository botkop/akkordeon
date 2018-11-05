package botkop.akkordeon.single

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import botkop.akkordeon.DataIterator
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
  val b2t: (Batch, ActorRef) => Forward = (b, sentinel) => {
    Forward(sentinel, b.x, b.y)
  }

  val b2v: (Batch, ActorRef) => Validate = (b, sentinel) => {
    Validate(sentinel, b.x, b.y)
  }

  def apply(dataSet: String,
            mode: String,
            miniBatchSize: Int,
            take: Option[Int] = None,
            name: String): DataProvider = {
    //val dl = DataLoader.instance(dataSet, mode, miniBatchSize, take)
    val dl = new MnistDataLoader(mode, miniBatchSize, take)
    val f = mode match {
      case "train" => b2t
      case "dev"   => b2v
    }
    DataProvider(dl, f, name)
  }
}

class DataProviderActor(dp: DataProvider) extends Actor with ActorLogging {

  import dp._

  override def receive: Receive = {
    val ndi = dl.iterator
    provide(ndi, ndi.next(), 1, System.currentTimeMillis())
  }

  def provide(di: DataIterator,
              nextBatch: (Variable, Variable),
              epoch: Int,
              startTime: Long): Receive = {
    case NextBatch(r) =>
      r forward f(Batch(nextBatch), sender())
      if (di.hasNext) {
        context become provide(di, di.next(), epoch, startTime)
      } else {
        val duration = System.currentTimeMillis() - startTime
        sender() ! Epoch(name, epoch, duration)

        val ndi = dl.iterator
        context become provide(ndi,
                               ndi.next(),
                               epoch + 1,
                               System.currentTimeMillis())
      }
  }
}
