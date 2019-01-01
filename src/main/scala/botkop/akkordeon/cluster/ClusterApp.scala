package botkop.akkordeon.cluster

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import akka.cluster.Cluster
import akka.routing.FromConfig
import botkop.akkordeon._
import com.typesafe.config.ConfigFactory
import scorch._
import scorch.autograd.Variable
import scorch.nn.{Linear, Module}
import scorch.optim.DCASGDa

import scala.language.postfixOps

object ClusterApp extends App {
  SentinelApp.main(Array("2551", "ts", "train"))
  SentinelApp.main(Array("2552", "vs", "validate"))
  NetworkApp.main(Array.empty)
}

object SentinelApp extends App {
  val batchSize = 1024
  val concurrency = 1
  val take = None

  val port = args(0)
  val name = args(1)
  val mode = args(2)

  val tdp = DataProvider("mnist", mode, batchSize, take, s"dp$name")

  val config = ConfigFactory
    .parseString(s"""
        akka.remote.artery.canonical.port=$port
        """)
    .withFallback(ConfigFactory.parseString("akka.cluster.roles = [sentinel]"))
    .withFallback(ConfigFactory.load("cluster"))

  implicit val system: ActorSystem = ActorSystem("AkkordeonCluster", config)
  Sentinel(tdp, concurrency, softmaxLoss, List(accuracy), "sentinel").stage
}

class NetworkActor extends Actor with ActorLogging {

  val backend: ActorRef =
    context.actorOf(FromConfig.props(), name = "sentinelRouter")

  val imageSize: Int = 28 * 28
  val batchSize = 1024

  val sizes = List(imageSize, 50, 20, 10)
  val learningRates = List(2e-2, 1e-2, 5e-3)
  val dropOuts = List(0.15, 0.07, 0.03)
  val gates: List[Gate] = makeNet(sizes, learningRates, dropOuts)
  val net: List[ActorRef] = Stageable.connect(gates)(context.system)

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
            val drop = DropConnect(dropOuts(i))
            def forward(x: Variable): Variable = x ~> fc ~> relu ~> drop
          }
          val o = DCASGDa(m.parameters, learningRates(i))
        Gate(m, o, s"g$i")
    } toList

  override def preStart(): Unit = {
    backend ! Wire(net)
  }

  override def receive: Receive = {
    case msg => log.info(msg.toString)
  }
}

object NetworkApp extends App {

  val config = ConfigFactory
    .parseString("akka.cluster.roles = [network]")
    .withFallback(ConfigFactory.load("cluster"))
  implicit val system: ActorSystem = ActorSystem("AkkordeonCluster", config)

  Cluster(system) registerOnMemberUp {
    system.actorOf(Props(classOf[NetworkActor]),
      name = "network")
  }

}
