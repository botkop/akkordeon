package botkop.akkordeon.mklstateless

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.nn.{Linear => MklLinear}
import com.intel.analytics.bigdl.tensor.Tensor

abstract class Module {
  lazy val (subModules, localParameters) =
    this.getClass.getDeclaredFields
      .foldLeft(List.empty[Module], List.empty[Variable]) {
        case ((zm, zv), f) =>
          f setAccessible true
          f get this match {
            case m: Module   => (m :: zm, zv)
            case v: Variable => (zm, v :: zv)
            case _           => (zm, zv)
          }
      }

  lazy val parameters: Seq[Variable] =
    localParameters ++ subModules.flatMap(_.parameters)

  lazy val gradients: Seq[Tensor[Float]] = parameters.map(_.g)

  def zeroGrad(): Unit = parameters.foreach(p => p.g.zero())
  def forward(x: Variable): Variable
  def apply(x: Variable): Variable = forward(x)
}

case class Linear(inputSize: Int, outputSize: Int) extends Module {
  val m = new MklLinear[Float](inputSize, outputSize)

  override def forward(x: Variable): Variable = ModuleFunction(x, m).forward()
}

case class ModuleFunction(x: Variable, m: TensorModule[Float]) extends Function {
  lazy val result: Tensor[Float] = {
    m.clearState()
    m.forward(x.data).clone()
  }

  override def forward(): Variable = Variable(result, Some(this))

  override def backward(g: Tensor[Float]): Unit = {
    m.clearState()
    m.output.set(result)
    val gi = m.updateGradInput(x.data, g).clone()
    x.backward(gi)
  }
}

