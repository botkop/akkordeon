package botkop.akkordeon.mklstateless

import com.intel.analytics.bigdl.tensor.Tensor

case class Variable(data: Tensor[Float],
                    f: Option[Function] = None,
                    name: Option[String] = None) {

  lazy val g: Tensor[Float] = data.clone().zero()

  def shape: List[Int] = data.size().toList

  def backward(gradOutput: Tensor[Float]): Unit = {
    g add gradOutput
    for (gf <- f) gf.backward(gradOutput)
  }

  def ~>(f: Variable => Variable): Variable = f(this)

  def t(): Variable = Function.transpose(this)
}
