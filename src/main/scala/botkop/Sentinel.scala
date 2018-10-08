package botkop

import akka.actor.Actor
import botkop.Gate.{Forward, Wiring}
import scorch.data.loader.DataLoader
import scorch.autograd.Function

class Sentinel(dl: DataLoader, loss: Function) extends Actor {

  override def receive: Receive = {
    case wire: Wiring =>
      context become forwardHandle(wire)
  }

  def forwardHandle(wiring: Wiring): Receive = {
    case Forward(v) =>

  }
}

object Sentinel {
  case object Start
}
