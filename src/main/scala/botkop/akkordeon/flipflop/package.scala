package botkop.akkordeon

import scorch.autograd.Variable

package object flipflop {

  case object Start
  case class Forward(x: Variable)
  case class Backward(g: Variable)
  case class Eval(x: Variable)

}
