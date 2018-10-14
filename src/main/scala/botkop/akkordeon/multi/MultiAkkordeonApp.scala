package botkop.akkordeon.multi

import akka.actor.ActorSystem
import scorch.data.loader.DataLoader
import scorch._
import botkop.akkordeon._
import botkop.multi.Start

object MultiAkkordeonApp extends App {

  implicit val system: ActorSystem = ActorSystem("akkordeon")
  val lr = 0.01
  val imageSize = 28 * 28
  val net = makeNet(lr, imageSize, 100, 10)

  val batchSize = 16
  val tdl: DataLoader = DataLoader.instance("mnist", "train", batchSize)
  val vdl: DataLoader = DataLoader.instance("mnist", "dev", batchSize)

  val ts = TrainingSentinel(tdl, softmaxLoss)
  val vs = ValidationSentinel(vdl, softmaxLoss, accuracy)

  val sentinels = List(ts, vs).map(_.stage)
  val gates = net.map(_.stage)

  MultiWire.wire(sentinels, gates)

  sentinels.foreach(_ ! Start)

}
