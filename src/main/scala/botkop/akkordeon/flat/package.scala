package botkop.akkordeon

import scorch.autograd.Variable

package object flat {
  case object Start
  case class Forward(x: Variable, y: Variable)
  case class Backward(g: Variable)
  case class Validate(x: Variable)

  case object NextBatch
  case class Batch(x: Variable, y: Variable)
  case object Batch {
    def apply(xy: (Variable, Variable)): Batch = Batch(xy._1, xy._2)
  }
  case class Epoch(n: Int, duration: Long)
}
