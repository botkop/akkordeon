package botkop

import org.scalatest.{FlatSpec, Matchers}

import scala.collection.immutable
import scala.util.Random

class SpeedSpec extends FlatSpec with Matchers {

  "random" should "next int vs next str" in {

    def nextStr = () => Random.nextString(4)
    measure(Random.nextInt)
    measure(nextStr)
  }

  def measure[T](f: () => T): Long = {
    val howMany = 1000000
    val t0 = System.nanoTime()
    var i = 0
    while (i < howMany) {
      val r = f()
      i += 1
    }
    val d0 = System.nanoTime() - t0
    println(d0)
    d0
  }

  it should "generate an alfanum string" in {
    val n = Random.alphanumeric.take(4).mkString
    println(n)
  }

}
