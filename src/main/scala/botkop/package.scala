import scorch.autograd.Variable
import scorch.nn.{Linear, Module}
import scorch.optim.SGD
import scorch._
import botkop.{numsca => ns}

import scala.language.postfixOps

package object botkop {

  type DataIterator = Iterator[(Variable, Variable)]

  def makeNet(lr: Double, sizes: Int*): List[Gate] = {
    sizes.sliding(2, 1).zipWithIndex.map { case (l, i) =>
      val m: Module = new Module() {
        val fc = Linear(l.head, l.last)
        override def forward(x: Variable): Variable = x ~> fc ~> relu
      }
      val o = SGD(m.parameters, lr)
      Gate(m, o, s"g$i")
    } toList
  }

  def accuracy(yHat: Variable, y: Variable): Double = {
    val guessed = ns.argmax(yHat.data, axis = 1)
    ns.mean(y.data == guessed).squeeze()
  }

}
