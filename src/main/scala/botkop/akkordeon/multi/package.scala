package botkop

import scorch.autograd.Variable

package object multi {
  case object Start
  case class Forward(v: Variable)
  case class Backward(v: Variable)
  case class Eval(x: Variable)
  case class EvalResult(loss: Double, eval: Double)
}
