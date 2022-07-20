import java.util.Locale

import org.scalatest.FunSuite
import org.scalactic._
import rl.utils.Utils

class UtilsTest extends FunSuite {
  implicit val doubleEquality: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(0.01)
  Locale.setDefault(Locale.US)

  test("Utils.getLogisticFunctionTest") {
    assert(Utils.getLogisticFunction(1.0)(0.0) === 0.5)
    assert(Utils.getLogisticFunction(-1.0)(math.log(9.0)) === 0.1)
  }

}
