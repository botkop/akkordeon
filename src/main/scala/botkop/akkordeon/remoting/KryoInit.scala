package botkop.akkordeon.remoting

import com.esotericsoftware.kryo.Kryo
import org.nd4j.Nd4jSerializer
import org.nd4j.linalg.api.ndarray.INDArray

class KryoInit {

  def customize(kryo: Kryo): Unit = {
    kryo.addDefaultSerializer(classOf[INDArray], classOf[Nd4jSerializer])
    kryo.register(classOf[INDArray])

    // add this when using idstrategy = incremental or explicit
    // kryo.addDefaultSerializer(classOf[org.nd4j.linalg.cpu.nativecpu.NDArray], classOf[Nd4jSerializer])
    // kryo.register(classOf[org.nd4j.linalg.cpu.nativecpu.NDArray])

  }

}
