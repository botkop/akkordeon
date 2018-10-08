package botkop

import akka.actor.ActorSystem
import scorch._
import scorch.autograd.Variable
import scorch.nn._
import scorch.optim.SGD

object Akkordeon extends App {

  implicit val system: ActorSystem = ActorSystem("akkordeon")

  val m1 = new Module() {
    val fc = Linear(32 * 32, 100)
    override def forward(x: Variable): Variable = x ~> fc ~> relu
  }
  val o1 = SGD(m1.parameters, 1e-3)
  val g1 = Gate(m1, o1)

  val m2 = new Module() {
    val fc = Linear(100, 10)
    override def forward(x: Variable): Variable = x ~> fc ~> relu
  }
  val o2 = SGD(m2.parameters, 1e-3)
  val g2 = Gate(m2, o2)

  val lgs = Gate.stage(g1, g2)

}

