package botkop.akkordeon

import scorch.autograd.Variable

package object flat {
  case object Start
  case class Forward(x: Variable, y: Variable)
  case class Backward(g: Variable)
  case class Validate(x: Variable, y: Variable)
}
