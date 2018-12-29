package botkop.akkordeon

import akka.actor.ActorSystem
import botkop.{numsca => ns}
import scorch._
import scorch.autograd.Variable
import scorch.nn.{Dropout, Linear, Module}
import scorch.optim._

import scala.language.postfixOps

object Akkordeon extends App {

  implicit val system: ActorSystem = ActorSystem("akkordeon")

  ns.rand.setSeed(231L)

  def makeNet(sizes: List[Int], lrs: List[Double], drops: List[Double]): List[Gate] =
    sizes
      .sliding(2, 1)
      .zipWithIndex
      .map {
        case (l, i) =>
          val m: Module = new Module() {
            val fc = Linear(l.head, l.last)
            // val drop = Dropout(drops(i))
            val drop = DropConnect(drops(i))
            def forward(x: Variable): Variable = x ~> fc ~> drop ~> relu
          }
          val o = DCASGDa(m.parameters, lrs(i))
        Gate(m, o, s"g$i")
    } toList

  def accuracy(yHat: Variable, y: Variable): Double = {
    val guessed = ns.argmax(yHat.data, axis = 1)
    ns.mean(y.data == guessed).squeeze()
  }

  def makeSentinels(takes: List[Option[Int]], concurrencies: List[Int]): Unit = {
    for (i <- concurrencies.indices) {
      val tdp = DataProvider("mnist", "train", batchSize, takes(i), s"tdp$i")
      val ts = Sentinel(tdp, concurrencies(i), softmaxLoss, List(accuracy), s"ts$i").stage
      ts ! Wire(Some(gates.last), Some(gates.head))
      ts ! Start
    }
  }

//  val sizes = List(28 * 28, 50, 20, 10)
//  val learningRates = List(2e-2, 1e-2, 5e-3)
//  val dropOuts = List(0.5, 0.2, 0.1)
//  val batchSize = 2048
/*
[info] [INFO] [12/27/2018 16:52:47.403] [akkordeon-akka.actor.default-dispatcher-5] [akka://akkordeon/user/ts2] tdp2       epoch:   466 loss:  0.015088 duration: 4732.425ms scores: (0.9979911843935648)
[info] [INFO] [12/27/2018 16:52:51.887] [akkordeon-akka.actor.default-dispatcher-3] [akka://akkordeon/user/ts1] tdp1       epoch:   467 loss:  0.014853 duration: 4772.769ms scores: (0.9983018000920614)
[info] [INFO] [12/27/2018 16:52:51.968] [akkordeon-akka.actor.default-dispatcher-11] [akka://akkordeon/user/ts0] tdp0       epoch:   234 loss:  0.027066 duration: 9800.093ms scores: (0.9945518096288045)
[info] [INFO] [12/27/2018 16:52:52.356] [akkordeon-akka.actor.default-dispatcher-12] [akka://akkordeon/user/ts2] tdp2       epoch:   467 loss:  0.015209 duration: 4750.799ms scores: (0.9980237364768982)
[info] [INFO] [12/27/2018 16:52:56.449] [akkordeon-akka.actor.default-dispatcher-12] [akka://akkordeon/user/ts1] tdp1       epoch:   468 loss:  0.014712 duration: 4376.478ms scores: (0.9982366959253947)
[info] [INFO] [12/27/2018 16:52:56.669] [akkordeon-akka.actor.default-dispatcher-5] [akka://akkordeon/user/vs1] ___VDP___  epoch:   112 loss:  0.161498 duration: 19910.04ms scores: (0.9656748235225677)
 */

  val sizes = List(28 * 28, 100, 40, 10)
  val learningRates = List(1e-2, 5e-3, 1e-3)
  val dropOuts = List(0.5, 0.2, 0.1)
  val batchSize = 2048

  val net = makeNet(sizes, learningRates, dropOuts)
  val gates = Stageable.connect(net)

  makeSentinels(List(None, Some(30000), Some(30000)), List(1, 1, 1))

  val vdp = DataProvider("mnist", "validate", 1024, None, "___VDP___")
  val vs = Sentinel(vdp, 1, softmaxLoss, List(accuracy), "vs").stage
  vs ! Wire(Some(gates.last), Some(gates.head))

  while (true) {
    Thread.sleep(60000)
    vs ! Start
  }

}
