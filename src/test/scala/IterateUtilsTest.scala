import java.util.Locale

import breeze.numerics._
import org.scalactic._
import org.scalatest.FunSuite
import rl.IterateUtils._

class IterateUtilsTest extends FunSuite {
  
  implicit val doubleEquality: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(0.0001)
  Locale.setDefault(Locale.US)
  
  val ns: LazyList[Int] = iterateLazyList((x: Int) => x + 1, 0)
  val ns2: Iterator[Int] = iterate((x: Int) => x + 1, 0)
  
  test("Test function iterate") {
    assert(ns.take(5).toList === (0 until 5).toList)
    assert(ns2.take(5).toList === (0 until 5).toList)
  }
  
  test("Test function last") {
    assert(last(0 until 5) === Some(4))
    assert(last(0 until 10) === Some(9))
    assert(last(Seq.empty) === None)
  }
  
  test("Test function converge") {
    
    val ns: LazyList[Double] = iterateLazyList((x: Int) => x + 1, 1).map(n => 1.0 / n.toDouble)
    def close(a: Double, b: Double): Boolean = abs(a - b) < 0.1
    val x = converged(ns, close)
    
    assert(x === 0.25)
  }
  
  test("Test function convergeV2") {
    
    val ns: Iterator[Double] = iterate((x: Int) => x + 1, 1).map(n => 1.0 / n.toDouble)
    def close(a: Double, b: Double): Boolean = abs(a - b) < 0.1
    val x = converged(ns, close)
    
    assert(x === 0.25)
  }
  
  test("Test converge end") {
    
    val ns = List(1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
    def close(a: Double, b: Double): Boolean = abs(a - b) < 0.1
    val x = converged(ns, close)
    
    assert(x === 2.0)
  }
  
}
