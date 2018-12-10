package botkop.akkordeon.mklseq

import com.intel.analytics.bigdl.tensor.Tensor

abstract class Optimizer[T](parameters: (Array[Tensor[T]], Array[Tensor[T]])) {

  val ps: Array[Tensor[T]] = parameters._1
  val dps: Array[Tensor[T]] = parameters._2

  def step(): Unit
  def zeroGrad(): Unit = dps.foreach(dp => dp.zero())
}
