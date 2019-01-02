package botkop.akkordeon.remoting

import java.util.UUID

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

object RemotingUtil {
  // some settings
  val clusterName = "AkkordeonCluster"
  val nameOfFirstGate = "gh" // head gate
  val nameOfLastGate = "gt" // tail gate

  def makeActorSystem(address: String): ActorSystem = {
    val Array(host, port) = address.split(":")
    makeActorSystem(host, port)
  }

  def makeActorSystem(host: String, port: String): ActorSystem = {
    val config = ConfigFactory
      .parseString(s"""
                      |akka.remote.artery.canonical.hostname = $host
                      |akka.remote.artery.canonical.port = $port
                      |""".stripMargin)
      .withFallback(ConfigFactory.load("remoting"))
    ActorSystem(RemotingUtil.clusterName, config)
  }

  def startSentinel(sentinel: ActorRef, nnAddress: String)(
      implicit system: ActorSystem): Unit = {

    implicit val timeout: Timeout = Timeout(3 seconds)
    implicit val execContext: ExecutionContextExecutor = system.dispatcher

    def path(gate: String): String =
      s"akka://$clusterName@$nnAddress/user/$gate"

    def resolveGate(gate: String): Future[ActorRef] =
      system
        .actorSelection(path(gate))
        .resolveOne()

    Future
      .sequence(List(resolveGate(nameOfFirstGate), resolveGate(nameOfLastGate)))
      .foreach {
        case List(head, last) =>
          sentinel ! Wire(last, head)
          sentinel ! Start
      }
  }

}

object RemotingApp extends App {
  NetworkApp.main(Array("127.0.0.1:25520"))
  SentinelApp.main(Array("127.0.0.1", "train", "1000", "127.0.0.1:25520"))
  SentinelApp.main(Array("127.0.0.1", "train", "3000", "127.0.0.1:25520"))
  SentinelApp.main(Array("127.0.0.1", "validate", "10000", "127.0.0.1:25520"))
}

object SentinelApp extends App {
  val localAddr = args(0)
  val mode = args(1)
  val take = Some(args(2).toInt)
  val nnAddress = args(3)

  val batchSize = 256
  val concurrency = 1
  val name = s"$mode-${take.get}-${UUID.randomUUID().toString}"

  implicit val system: ActorSystem =
    RemotingUtil.makeActorSystem(localAddr, "0")

  val tdp = DataProvider("mnist", mode, batchSize, take, s"dp4$name")
  val sentinel =
    Sentinel(tdp, concurrency, softmaxLoss, List(accuracy), name).stage

  RemotingUtil.startSentinel(sentinel, nnAddress)

  if (mode == "validate") {
    while (true) {
      Thread.sleep(10000)
      sentinel ! Start
    }
  }
}

object NetworkApp extends App {
  val localAddr = args(0)

  implicit val system: ActorSystem = RemotingUtil.makeActorSystem(localAddr)

  val imageSize: Int = 28 * 28
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
          val name = {
            if (i == 0) RemotingUtil.nameOfFirstGate
            else if (i >= sizes.length - 2) RemotingUtil.nameOfLastGate
            else s"g$i"
          }
          Gate(m, o, name)
      }
      .toList

}
