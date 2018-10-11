package botkop

import akka.actor.ActorSystem
import botkop.Sentinel.Start
import scorch._
import scorch.data.loader.DataLoader

import scala.language.postfixOps

object Akkordeon extends App {

  implicit val system: ActorSystem = ActorSystem("akkordeon")

  val lr = 0.01
  val imageSize: Int = 28 * 28

  val net = makeNet(lr, imageSize, 50, 20, 10)

  val batchSize = 16
  val tdl: DataLoader = DataLoader.instance("mnist", "train", batchSize)
  val vdl: DataLoader = DataLoader.instance("mnist", "dev", batchSize)

  val s = Sentinel(tdl, vdl, softmaxLoss, accuracy, "sentinel")
  val ring = Wiring.wire(s, net)

  ring.head ! Start
}

