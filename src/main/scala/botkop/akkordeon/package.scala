package botkop

import scorch.autograd.Variable

import scala.language.postfixOps

package object akkordeon {
  type DataIterator = Iterator[(Variable, Variable)]
}
