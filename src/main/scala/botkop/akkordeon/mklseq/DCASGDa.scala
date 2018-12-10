package botkop.akkordeon.mklseq

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
/**
  * Asynchronous Stochastic Gradient Descent with Delay Compensation
  * with adaptive lambda
  * https://arxiv.org/abs/1609.08326
  */
case class DCASGDa[T](parameters: (Array[Tensor[T]], Array[Tensor[T]]),
              lr: Double,
              momentum: Double = 0.95,
              lambda: Double = 2)(implicit ev: TensorNumeric[T])
  extends Optimizer(parameters) {


  val previousWeights: Array[Tensor[T]] = parameters._1.map { v =>
    v.clone()
  }
  val meanSquare: Array[Tensor[T]] = parameters._1.map { v =>
    v.clone().zero()
  }
  val epsilon = 1e-7

  override def step(): Unit =
    parameters._1.indices.foreach { i =>
      val weight = parameters._1(i)
      val grad = parameters._2(i)
      val previousWeight = previousWeights(i)
      val ms = meanSquare(i)

      /*
      meanSquare = (meanSquare * momentum) + (1 - momentum) * grad * grad
       */
      val cg = grad.clone()
      cg.cmul(grad)
      cg.mul(ev.fromType(1.0 - momentum))
      ms.mul(ev.fromType(momentum))
      ms.add(cg)

      // val upd = -lr * (grad + (lambda / ns.sqrt(meanSquare(i) + epsilon)) * grad * grad * (weight - previousWeight))

      // (lambda / ns.sqrt(meanSquare(i) + epsilon)
      val upd = ms + ev.fromType(epsilon)
      upd.sqrt()
      upd.pow(ev.fromType(-1.0))
      upd.mul(ev.fromType(lambda))

      upd.cmul(grad)
      upd.cmul(grad)
      val wm = weight - previousWeight
      upd.cmul(wm)

      upd.add(grad)

      upd.mul(ev.fromType(-lr))

      previousWeight.copy(weight)

      // weight += upd
      weight.add(upd)

    }
}
