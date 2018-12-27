package botkop.akkordeon.hash

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import scorch.autograd.{Function, Variable}
import scorch.nn.Module

case class DropConnect(p: Double = 0.5) extends Module {
  import DropConnect._
  override def forward(x: Variable): Variable =
    DropConnectFunction(x, p, inTrainingMode).forward()
}

object DropConnect {

  case class DropConnectFunction(x: Variable,
                                 p: Double = 0.5,
                                 train: Boolean = false)
      extends Function {

    val mask: Tensor = ns.rand(x.shape: _*) < p

    override def forward(): Variable =
      if (train)
        Variable(x.data * mask, Some(this))
      else
        Variable(x.data, Some(this))

    override def backward(gradOutput: Variable): Unit =
      if (train)
        x.backward(Variable(gradOutput.data * mask))
      else
        x.backward(gradOutput)

  }

}
