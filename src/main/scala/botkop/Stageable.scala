package botkop

import akka.actor.{ActorRef, ActorSystem}

trait Stageable {
  def stage(implicit system: ActorSystem): ActorRef
}
