package botkop.akkordeon.remoting

import akka.actor.{ActorRef, ActorSystem}
import akka.util.Timeout
import botkop.akkordeon._
import com.typesafe.config.ConfigFactory
import scorch._
import scorch.autograd.Variable
import scorch.nn.{Dropout, Linear, Module}
import scorch.optim.DCASGDa

import scala.concurrent.duration._
import scala.concurrent.{ExecutionContextExecutor, Future}
import scala.language.postfixOps

object RemotingSettings {
  // some settings
  val clusterName = "AkkordeonCluster"
  val numTrainingSamples = 60000
  val numValidationSamples = 10000
  val nameOfFirstGate = "g0"
  val nameOfLastGate = "g2"
}

object RemotingApp extends App {
  NetworkApp.main(Array("127.0.0.1:25520"))

  SentinelApp.main(
    Array("127.0.0.1:25521",
          "trainingSentinel1",
          "train",
          "60000",
          "127.0.0.1:25520"))

  SentinelApp.main(
    Array("127.0.0.1:25522",
          "trainingSentinel2",
          "train",
          "3000",
          "127.0.0.1:25520"))

}

object SentinelApp extends App {
  val localAddr = args(0)
  val name = args(1)
  val mode = args(2)
  val take = Some(args(3).toInt)
  val nnAddress = args(4)

  val batchSize = 256
  val concurrency = 1

  val Array(host, port) = localAddr.split(":")
  val config = ConfigFactory
    .parseString(s"""
                    |akka.remote.artery.canonical.hostname = $host
                    |akka.remote.artery.canonical.port = $port
                    |""".stripMargin)
    .withFallback(ConfigFactory.load("remoting"))

  implicit val system: ActorSystem =
    ActorSystem(RemotingSettings.clusterName, config)

  val tdp = DataProvider("mnist", mode, batchSize, take, s"dp4$name")
  val sentinel =
    Sentinel(tdp, concurrency, softmaxLoss, List(accuracy), name).stage

  implicit val execContext: ExecutionContextExecutor = system.dispatcher
  implicit val timeout: Timeout = Timeout(3 seconds)

  def path(gate: String) =
    s"akka://${RemotingSettings.clusterName}@$nnAddress/user/$gate"

  Future
    .sequence(
      List(
        system
          .actorSelection(path(RemotingSettings.nameOfFirstGate))
          .resolveOne(),
        system
          .actorSelection(path(RemotingSettings.nameOfLastGate))
          .resolveOne()
      ))
    .foreach {
      case List(head, last) =>
        sentinel ! Wire(last, head)
        sentinel ! Start
    }
}

object NetworkApp extends App {
  val localAddr = args(0)
  val Array(host, port) = localAddr.split(":")
  val config = ConfigFactory
    .parseString(s"""
         |akka.remote.artery.canonical.hostname = $host
         |akka.remote.artery.canonical.port = $port
         |""".stripMargin)
    .withFallback(ConfigFactory.load("remoting"))
  implicit val system: ActorSystem = ActorSystem("AkkordeonCluster", config)

  val imageSize: Int = 28 * 28
  val batchSize = 1024
  val sizes = List(imageSize, 50, 20, 10)
  val learningRates = List(2e-2, 1e-2, 5e-3)
  val dropOuts = List(0.15, 0.07, 0.03)
  val gates: List[Gate] = makeNet(sizes, learningRates, dropOuts)
  val net: List[ActorRef] = Stageable.connect(gates)

  def makeNet(sizes: List[Int],
              learningRates: List[Double],
              dropOuts: List[Double]): List[Gate] =
    sizes
      .sliding(2, 1)
      .zipWithIndex
      .map {
        case (l, i) =>
          val m: Module = new Module() {
            val fc = Linear(l.head, l.last)
            val drop = Dropout(dropOuts(i))
            def forward(x: Variable): Variable = x ~> fc ~> relu ~> drop
          }
          val o = DCASGDa(m.parameters, learningRates(i))
        Gate(m, o, s"g$i")
    } toList

}
