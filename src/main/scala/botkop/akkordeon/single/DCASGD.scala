package botkop.akkordeon.single

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import scorch.autograd.Variable
import scorch.optim.Optimizer

/**
  * Asynchronous Stochastic Gradient Descent with Delay Compensation
  * https://arxiv.org/abs/1609.08326
  */
case class DCASGD(parameters: Seq[Variable],
                  lr: Double,
                  wd: Double = 0.0,
                  useMomentum: Boolean = false,
                  momentum: Double = 0.9,
                  lambda: Double = 0.04)
    extends Optimizer(parameters) {


  val previousWeights: Seq[Tensor] = parameters.map { v =>
    v.data.copy()
  }
  val momenta: Seq[Option[Tensor]] = parameters.map { v =>
    if (useMomentum) Some(ns.zerosLike(v.data)) else None
  }

//  val mc = 0.2345 // m is a constant taking value from [0, 1)
//  val epsilon = 1e-7
//  val meanSquare: Seq[Tensor] = parameters.map { v =>
//    ns.zerosLike(v.data)
//  }

  var t = 0

  override def step(): Unit = {
    t += 1

    parameters.indices.foreach { i =>
      val weight = parameters(i).data
      val grad = parameters(i).grad.data
      val previousWeight = previousWeights(i)
      val maybeMomentum = momenta(i)

//      val ms = meanSquare(i)
//      ms := mc * ms + (1 - mc) * grad * grad
//      val l1 = lambda / ns.sqrt(ms + epsilon)
//      val upd =
//        -lr * (grad + (wd * weight) + l1 * grad * grad * (weight - previousWeight))

      val upd =
        -lr * (grad + (wd * weight) + lambda * grad * grad * (weight - previousWeight))

      val mom = maybeMomentum match {
        case None => upd
        case Some(m) =>
          m *= momentum
          m += upd
          m
      }

      previousWeight := weight
      weight += mom
    }
  }
}
