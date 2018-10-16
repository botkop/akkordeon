package botkop

import botkop.akkordeon.flipflop.Gate
import botkop.{numsca => ns}
import scorch._
import scorch.autograd.Variable
import scorch.nn.{Linear, Module}
import scorch.optim.SGD

import scala.language.postfixOps

package object akkordeon {
  type DataIterator = Iterator[(Variable, Variable)]

}
