import java.util.Locale

import breeze.linalg.DenseVector
import org.scalactic._
import org.scalatest.FunSuite
import rl._
import rl.utils.MapAlgebra._

class FunctionApproximationTest extends FunSuite {
  
  implicit val doubleEquality: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(0.0001)
  Locale.setDefault(Locale.US)
  
  val map0 = Map(0 -> 0.0, 1 -> 0.0, 2 -> 0.0)
  val mapAlmost0 = Map(0 -> 0.01, 1 -> 0.01, 2 -> 0.01)
  val map1 = Map(0 -> 1.0, 1 -> 2.0, 2 -> 3.0)
  val mapAlmost1 = Map(0 -> 1.01, 1 -> 2.01, 2 -> 3.01)
  val map2 = Map(0 -> 1.0, 1 -> 1.0, 2 -> 1.0)
  
  val dynamic0: Dynamic[Int] = Dynamic(map0)
  val dynamicAlmost0: Dynamic[Int] = Dynamic(mapAlmost0)
  
  val dynamic1: Dynamic[Int] = Dynamic(map1)
  val dynamicAlmost1: Dynamic[Int] = Dynamic(mapAlmost1)
  
  test("Test Matrix Algebra"){
    val testMap1 = sumMap(map0, map1)
    val testMap2 = multiplyByScalar(map0, 2.0)
    val testMap3 = multiplyByScalar(map1, 2.0)
    
    val testMap4 = productMap(map0, map1)
    val testMap5 = productMap(map2, map1)
    val testMap6 = productMap(map1, map1)
    
    assert(testMap1 === map1)
    assert(testMap2 === map0)
    assert(testMap3 === Map(0 -> 2.0, 1 -> 4.0, 2 -> 6.0))
    assert(testMap4 === map0)
    assert(testMap5 === map1)
    assert(testMap6 === Map(0 -> 1.0, 1 -> 4.0, 2 -> 9.0))
  }
  
  test("Test update") {
    val updated = dynamic0.update(Seq(0 -> 1.0, 1 -> 2.0, 2 -> 3.0))
    val partiallyUpdated = dynamic0.update(Seq(1 -> 3.0))
    val repeatedUpdated = dynamic0.update(Seq(1 -> 2.0, 1 -> 3.0, 1 -> 2.0))
    
    assert(updated.valuesMap === dynamic1.valuesMap)
    assert(partiallyUpdated.valuesMap === Map(0 -> 0.0, 1 -> 3.0, 2 -> 0.0))
    assert(repeatedUpdated.valuesMap === Map(0 -> 0.0, 1 -> 2.0, 2 -> 0.0))
  }
  
  test("Test within") {
    assert(dynamic0.within(dynamic0, 0.0))
    assert(dynamic0.within(dynamicAlmost0, 0.011))
    assert(!dynamic0.within(dynamic1, 0.011))
    assert(dynamic1.within(dynamic1, 0.00))
    assert(dynamic1.within(dynamicAlmost1, 0.011))
  }
  
  test("Test apply") {
    assert((0 until 3).forall { i =>
      dynamic0(i) == 0.0
    })
    assert((0 until 3).forall { i =>
      dynamic1(i) == i + 1
    })
  }
  
  test("Test evaluate") {
    assert(dynamic0.evaluate(0 until 3) === DenseVector.zeros[Double](3))
    assert(dynamic1.evaluate(0 until 3) === DenseVector[Double](1, 2, 3))
  }
  
  test("Test Functional Vector Space operations"){
    val testMap1 = Map(0 -> 2.0, 1 -> 4.0, 2 -> 6.0)
    
    assert((dynamic0 * 3).valuesMap === dynamic0.valuesMap)
    assert((dynamic0 + dynamic1).valuesMap === dynamic1.valuesMap)
    assert((dynamic1 * 2).valuesMap === testMap1)
    assert((dynamicAlmost0 + dynamic1).valuesMap === dynamicAlmost1.valuesMap)
  }
  
}
