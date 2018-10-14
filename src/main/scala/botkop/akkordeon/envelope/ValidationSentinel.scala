package botkop.akkordeon.envelope

import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import botkop.akkordeon.Stageable
import scorch.autograd.Variable
import scorch.data.loader.DataLoader

case class ValidationSentinel(dl: DataLoader,
                              loss: (Variable, Variable) => Variable,
                              evaluator: (Variable, Variable) => Double,
                              name: String)
    extends Stageable {
  override def stage(implicit system: ActorSystem): ActorRef =
    system.actorOf(Props(new ValidationSentinelActor(this)),
                   "validationSentinel")
}

class ValidationSentinelActor(sentinel: ValidationSentinel) extends Actor {
  override def receive: Receive = ???
}
