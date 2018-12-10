package botkop.akkordeon.mklstateless

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.tensor.Tensor

trait Function {
  def forward(): Variable
  def backward(g: Tensor[Float]): Unit
}

object Function {
  def transpose(v: Variable): Variable = Transpose(v).forward()
}

abstract class FunctionFactory(v: Variable) extends Function {
  def m: AbstractModule[Tensor[_], Tensor[_], Float]

  override def forward(): Variable = {
    val output = m.updateOutput(v.data).toTensor[Float]
    Variable(output, Some(this))
  }

  override def backward(g: Tensor[Float]): Unit = {
    val gradInput = m.updateGradInput(v.data, g).toTensor[Float]
    v.backward(gradInput)
  }
}

case class Transpose(v: Variable) extends FunctionFactory(v) {
  val m = new nn.Transpose[Float](Array((1, 2)))
}

