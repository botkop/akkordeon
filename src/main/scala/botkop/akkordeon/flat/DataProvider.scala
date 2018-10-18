package botkop.akkordeon.flat

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import botkop.akkordeon.{DataIterator, Stageable}
import scorch.autograd.Variable
import scorch.data.loader.DataLoader

case class DataProvider(dl: DataLoader, name: String) extends Stageable {
  override def stage(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new DataProviderActor(this)), name)
}

class DataProviderActor(dp: DataProvider) extends Actor with ActorLogging {

  override def receive: Receive = {
    val ndi = dp.dl.iterator
    provide(ndi, ndi.next(), 1, System.currentTimeMillis())
  }

  def provide(di: DataIterator, nextBatch: (Variable, Variable), epoch: Int, startTime: Long): Receive = {
    case NextBatch =>
      sender() ! Batch(nextBatch)
      if (di.hasNext) {
        context become provide(di, di.next(), epoch, startTime)
      } else {
        val duration = System.currentTimeMillis() - startTime
        // log.info(f"epoch: $epoch%5d duration: ${duration}ms")
        sender() ! Epoch(epoch, duration)

        val ndi = dp.dl.iterator
        context become provide(ndi, ndi.next(), epoch + 1, System.currentTimeMillis())
      }
  }
}
