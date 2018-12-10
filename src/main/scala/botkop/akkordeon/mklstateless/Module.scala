package botkop.akkordeon.mklstateless

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

case class Linear(weights: Variable, bias: Variable) extends Module {
  override def forward(x: Variable): Variable =
    x.dot(weights.t()) + bias
}
