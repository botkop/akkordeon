import scorch.autograd.Variable

package object botkop {

  type DataIterator = Iterator[(Variable, Variable)]
}
