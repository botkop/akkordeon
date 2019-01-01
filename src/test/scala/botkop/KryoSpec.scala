package botkop

import botkop.numsca.Tensor
import org.scalatest.{FlatSpec, Matchers}
import botkop.{numsca => ns}
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import org.nd4j.linalg.api.ndarray.INDArray
import org.objenesis.strategy.StdInstantiatorStrategy

class KryoSpec extends FlatSpec with Matchers {

  "kryo" should "serialize tensor" in {

    val kryo = new Kryo
    kryo.addDefaultSerializer(classOf[INDArray], classOf[org.nd4j.Nd4jSerializer])
    kryo.register(classOf[Tensor])
    kryo.setInstantiatorStrategy(new Kryo.DefaultInstantiatorStrategy(new StdInstantiatorStrategy()))

    val t = ns.zeros(2,2)

    import java.io.FileOutputStream
    val output = new Output(new FileOutputStream("file.bin"))
    kryo.writeObject(output, t)
    output.close()

    import java.io.FileInputStream
    val input = new Input(new FileInputStream("file.bin"))
    val object2 = kryo.readObject(input, classOf[Tensor])
    input.close()

    println(object2)

  }

}
