package botkop

import akka.actor.{Actor, ActorRef, Props}
import botkop.ModuleAktor.{Backward, Forward}
import scorch.autograd.Variable
import scorch.nn.Module
import scorch.optim.Optimizer

class ModuleAktor(module: Module,
                  optimizer: Optimizer,
                  prev: Option[ActorRef],
                  next: Option[ActorRef])
    extends Actor {

  override def receive: Receive = forwardHandle()

  def forwardHandle(): Receive = {
    case Forward(v) =>
      val result = module(v)
      for (n <- next) n ! Forward(result)
      context become backwardHandle(result)
  }

  def backwardHandle(v: Variable): Receive = {
    case Backward(g) =>
      optimizer.zeroGrad()
      v.backward(g)
      optimizer.step()
      for (p <- prev) p ! Backward(v)
      context become forwardHandle()
  }

}

object ModuleAktor {
  def props(module: Module,
            optimizer: Optimizer,
            prev: Option[ActorRef],
            next: Option[ActorRef]): Props =
    Props(new ModuleAktor(module, optimizer, prev, next))

  case class Forward(v: Variable)
  case class Backward(v: Variable)
}
